// FrictionEstimator.h

#pragma once
#include <array>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>

// ... 你原本的 FrictionEstimator 類別 ...

class FrictionEstimator
{
public:
    static constexpr int INPUT_DIM  = 2850; // 50 * 57
    static constexpr int OUTPUT_DIM = 12;

    explicit FrictionEstimator(const std::string& onnx_path);

    void reset();

    std::array<float, OUTPUT_DIM> infer(const std::vector<float>& x);

private:
    std::string onnx_path_;
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    std::string input_name_;
    std::string output_name_;

    std::vector<float> mean_;
    std::vector<float> std_;
};

// ====== 在這裡新增全域共享變數宣告 ======
extern std::array<float, FrictionEstimator::OUTPUT_DIM> g_mu_hat;
extern bool g_mu_valid;

