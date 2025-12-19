#include "FSM/FrictionEstimator.h"

#include <stdexcept>
#include <numeric>   // optional
#include <spdlog/spdlog.h>
#include <cmath>
#include <fstream>
#include <filesystem>

// ====== 全域變數定義（這裡真的配置記憶體） ======
std::array<float, FrictionEstimator::OUTPUT_DIM> g_mu_hat = {0.0f};
bool g_mu_valid = false;

static std::vector<float> load_f32_bin(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("[FrictionEstimator] Cannot open file: " + path);
    }
    ifs.seekg(0, std::ios::end);
    const std::streamsize bytes = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    if (bytes <= 0 || (bytes % static_cast<std::streamsize>(sizeof(float)) != 0)) {
        throw std::runtime_error("[FrictionEstimator] Bad float32 file size: " + path);
    }

    std::vector<float> v(static_cast<size_t>(bytes / sizeof(float)));
    if (!ifs.read(reinterpret_cast<char*>(v.data()), bytes)) {
        throw std::runtime_error("[FrictionEstimator] Failed to read file: " + path);
    }
    return v;
}

FrictionEstimator::FrictionEstimator(const std::string& onnx_path)
    : onnx_path_(onnx_path),
      env_(ORT_LOGGING_LEVEL_WARNING, "friction_estimator"),
      session_options_{},
      session_(nullptr)
{
    // 基本 session 設定
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // 建立 Session
    session_ = Ort::Session(env_, onnx_path_.c_str(), session_options_);

    // 取得 input / output 名稱 (1.22 版不需要 allocator，直接回傳 vector<string>)
    auto input_names  = session_.GetInputNames();
    auto output_names = session_.GetOutputNames();

    if (input_names.empty() || output_names.empty()) {
        throw std::runtime_error("[FrictionEstimator] ONNX model has no inputs/outputs.");
    }

    // 以第一個 input/output 為主
    input_name_  = input_names[0];   // std::string
    output_name_ = output_names[0];  // std::string

    spdlog::info("[FrictionEstimator] Loaded ONNX model: {}", onnx_path_);
    spdlog::info("[FrictionEstimator] Input name  : {}", input_name_);
    spdlog::info("[FrictionEstimator] Output name : {}", output_name_);

    // 如果你之後要加 normalization，可以在這裡讀 mean_/std_
    // 目前先留空，不影響推論
    try {
    namespace fs = std::filesystem;
    fs::path onnx_p(onnx_path_);
    fs::path dir = onnx_p.parent_path();

    fs::path mean_path = dir / "estimator_mean.bin";
    fs::path std_path  = dir / "estimator_std.bin";

    mean_ = load_f32_bin(mean_path.string());
    std_  = load_f32_bin(std_path.string());

    if (mean_.size() != INPUT_DIM || std_.size() != INPUT_DIM) {
        throw std::runtime_error(
            "[FrictionEstimator] mean/std size mismatch. mean=" + std::to_string(mean_.size()) +
            " std=" + std::to_string(std_.size()) +
            " expected=" + std::to_string(INPUT_DIM)
        );
    }

    spdlog::info("[FrictionEstimator] Loaded mean/std: mean={} std={}", mean_.size(), std_.size());
    spdlog::info("[FrictionEstimator] mean/std path: {} | {}", mean_path.string(), std_path.string());
} catch (const std::exception& e) {
    // If files missing, we keep mean_/std_ empty => no normalization (but output will likely be bad)
    spdlog::warn("[FrictionEstimator] No mean/std loaded (normalization disabled): {}", e.what());
    mean_.clear();
    std_.clear();
}
}

void FrictionEstimator::reset()
{
   g_mu_valid = false; 
}

std::array<float, FrictionEstimator::OUTPUT_DIM>
FrictionEstimator::infer(const std::vector<float>& x)
{
    if (static_cast<int>(x.size()) != INPUT_DIM) {
        spdlog::warn(
            "[FrictionEstimator] infer(): input size mismatch. Got {}, expect {}.",
            x.size(), INPUT_DIM
        );
    }

    // 建一份可修改的 input buffer
    std::vector<float> input = x;

    // 如果有 mean_ / std_ 就做 normalization
    if (!mean_.empty() && !std_.empty() &&
        mean_.size() == input.size() && std_.size() == input.size())
    {
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = (input[i] - mean_[i]) / (std_[i] + 1e-6f);
        }
    }

    // ==== 建立 ONNX Runtime tensor ====
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::array<int64_t, 2> input_shape{1, INPUT_DIM};  // (1, 2850)
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info,
        input.data(),
        input.size(),
        input_shape.data(),
        input_shape.size()
    );

    const char* input_names[]  = { input_name_.c_str() };
    const char* output_names[] = { output_name_.c_str() };

    // ==== 執行推論 ====
    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1
    );

    if (output_tensors.empty()) {
        throw std::runtime_error("[FrictionEstimator] session.Run() returned empty outputs.");
    }

    float* out_data = output_tensors[0].GetTensorMutableData<float>();

    std::array<float, OUTPUT_DIM> out{};
    for (int i = 0; i < OUTPUT_DIM; ++i) {
        out[i] = out_data[i];
    }

    return out;
}

