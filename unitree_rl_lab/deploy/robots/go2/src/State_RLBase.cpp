#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "FSM/FrictionEstimator.h"
#include <algorithm>  // std::clamp
#include <vector>
#include <cmath>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <spdlog/spdlog.h>

// ====== Estimator 相關常數 ======
namespace {
    constexpr int POLICY_HZ   = 100;
    constexpr int EST_HZ      = 10;
    constexpr int EST_STRIDE  = POLICY_HZ / EST_HZ;   // 10

    constexpr int HISTORY_LEN = 50;
    constexpr int FEAT_DIM    = 57;
    constexpr int INPUT_DIM   = HISTORY_LEN * FEAT_DIM;

    static std::array<float, FEAT_DIM> g_history[HISTORY_LEN];
    static int g_step_count   = 0;   // 目前收了多少步（<=50）
    static int g_policy_step  = 0;   // 用來做 10Hz stride

    static std::array<float, 12> g_last_action = {0.0f};

    // 用單一 estimator（由 ctor 用 policy_dir 的 onnx 初始化）
    static std::unique_ptr<FrictionEstimator> g_estimator;

    // ====== 重要：forward declaration + definition 要一致 ======
    static std::array<float, FEAT_DIM>
    build_estimator_feature(isaaclab::ManagerBasedRLEnv* env);
    
    static std::array<float, 12> g_mu_raw{};
    static bool g_mu_raw_valid = false;

    inline float clampf(float v, float lo, float hi) {
        return std::max(lo, std::min(v, hi));
    }

    static void maybe_infer_and_update_mu(const std::vector<float>& x_flat) {
        if (!g_estimator) return;

        auto mu_new = g_estimator->infer(x_flat);
        g_mu_raw_valid = true;
        for (int i=0;i<12;++i) g_mu_raw[i] = mu_new[i];

        // clamp：避免 estimator 偶發爆值害真機暴衝/跌倒
        for (int leg = 0; leg < 4; ++leg) {
            mu_new[3 * leg + 0] = clampf(mu_new[3 * leg + 0], 0.05f, 2.0f); // mu_s
            mu_new[3 * leg + 1] = clampf(mu_new[3 * leg + 1], 0.05f, 2.0f); // mu_d
            mu_new[3 * leg + 2] = clampf(mu_new[3 * leg + 2], 0.00f, 0.8f); // e
        }

        // 低通：10Hz 更新時很重要，避免 mu 跳動造成 policy 抖
        const float alpha = 0.6f; // 越大越平滑
        if (!g_mu_valid) {
            g_mu_hat = mu_new;
            g_mu_valid = true;
        } else {
            for (int i = 0; i < 12; ++i) {
                g_mu_hat[i] = alpha * g_mu_hat[i] + (1.0f - alpha) * mu_new[i];
            }
        }
    }

    static bool update_friction_estimator(isaaclab::ManagerBasedRLEnv* env) {
    // 每 step 都累加（只留這一次）
    g_policy_step++;

    // 1) 每步都更新 history
    auto feat = build_estimator_feature(env);

    if (g_step_count == 0) {
        for (int k = 0; k < HISTORY_LEN; ++k) g_history[k] = feat;
        g_step_count = 1;
    } else if (g_step_count < HISTORY_LEN) {
        g_history[g_step_count++] = feat;
    } else {
        for (int k = 0; k < HISTORY_LEN - 1; ++k) g_history[k] = g_history[k + 1];
        g_history[HISTORY_LEN - 1] = feat;
    }

    // history 未滿 50：不推論
    if (g_step_count < HISTORY_LEN) return false;

    // 2) 只有每 10 步推論一次（10Hz）
    if ((g_policy_step % EST_STRIDE) != 0) return false;

    std::vector<float> x(INPUT_DIM);
    int p = 0;
    for (int k = 0; k < HISTORY_LEN; ++k)
        for (int j = 0; j < FEAT_DIM; ++j)
            x[p++] = g_history[k][j];

    maybe_infer_and_update_mu(x);
    return true;
}

    // ====== 真正的定義（必須在同一個 namespace{}） ======
    static std::array<float, FEAT_DIM>
build_estimator_feature(isaaclab::ManagerBasedRLEnv* env) {
    std::array<float, FEAT_DIM> feat{};
    int idx = 0;

    // 1) 直接拿 observation_manager 的 "obs"
    //    因為 deploy.yaml 是 single-group 格式，ObservationManager 會把它命名為 "obs"
    const std::vector<float> obs = env->observation_manager->compute_group("obs");

    // obs 應該是 57 = 45(estimator obs) + 12(foot_friction)
    if (static_cast<int>(obs.size()) < 45) {
        spdlog::error("obs dim too small: got {}, expected at least 45", obs.size());
        return feat;
    }

    // 2) 取前 45 維（這就是訓練 estimator 用的 est_obs）
    for (int i = 0; i < 45; ++i) feat[idx++] = obs[i];

    // 3) 外加一次 last_action（照你的 collect script：feat_t=[est_obs,last_action]）
    for (int i = 0; i < 12; ++i) feat[idx++] = g_last_action[i];

    if (idx != FEAT_DIM) {
        spdlog::error("Estimator feature dim mismatch: idx={} FEAT_DIM={}", idx, FEAT_DIM);
    }
    return feat;
}
    // ===== CSV logger =====
    static std::ofstream g_csv;
    static bool g_csv_inited = false;
    static std::chrono::steady_clock::time_point g_t0;

    static double now_s() {
        using namespace std::chrono;
        return duration<double>(steady_clock::now() - g_t0).count();
    }

    static void init_csv_once(const std::string& path) {
        if (g_csv_inited) return;
        g_t0 = std::chrono::steady_clock::now();
        g_csv.open(path, std::ios::out);
        if (!g_csv.is_open()) {
            spdlog::error("Failed to open CSV log: {}", path);
            return;
        }

        // header
        g_csv << "t_wall,policy_step,est_updated";
        for (int i = 0; i < 12; ++i) g_csv << ",mu_hat_" << i;
        for (int i = 0; i < 12; ++i) g_csv << ",mu_hat_raw_" << i;
        g_csv << ",pg_x,pg_y,pg_z";
        g_csv << ",base_ang_vel_x,base_ang_vel_y,base_ang_vel_z";
        g_csv << ",cmd_vx,cmd_vy,cmd_wz";
        for (int i = 0; i < 12; ++i) g_csv << ",action_" << i;
        g_csv << "\n";

        g_csv << std::fixed << std::setprecision(6);
        g_csv_inited = true;
        spdlog::info("CSV log enabled: {}", path);
    }

    static void log_csv_row(
        isaaclab::ManagerBasedRLEnv* env,
        int policy_step,
        bool est_updated,
        const std::array<float, 12>& action_now
    ) {
        if (!g_csv_inited || !g_csv.is_open()) return;

        // 取 robot state
        auto& data = env->robot->data;

        // 你可能有不同的姿態表示法：下面假設你有 rpy 或 quaternion
        // 若你目前只有 projected_gravity + ang_vel，也可以先用 projected_gravity 代替 roll/pitch 的佐證
        // 這裡示範：假設 data.root_rpy_b (3) 存在；若沒有，跟我說你資料欄位名稱我幫你換。
        float roll = 0.f, pitch = 0.f, yaw = 0.f;
        if constexpr (true) {
            // TODO: 依你的實際欄位名稱替換
            // 可能是 data.root_rpy_w / data.root_rpy_b / data.base_rpy ...
            // 如果沒有這個欄位，先保留 0，不影響 mu 的主要驗證
        }

        // base_ang_vel_b (3) 你在 feature 生成時用過：data.root_ang_vel_b[i]
        float wx = data.root_ang_vel_b[0];
        float wy = data.root_ang_vel_b[1];
        float wz = data.root_ang_vel_b[2];

        // command（你 feature 裡也有計算過）
        float cmd_vx = 0.f, cmd_vy = 0.f, cmd_wz = 0.f;
        {
            auto& joystick = env->robot->data.joystick;
            const auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];
            cmd_vx = std::clamp(joystick->ly(),
                                cfg["lin_vel_x"][0].as<float>(),
                                cfg["lin_vel_x"][1].as<float>());
            cmd_vy = std::clamp(joystick->lx(),
                                cfg["lin_vel_y"][0].as<float>(),
                                cfg["lin_vel_y"][1].as<float>());
            cmd_wz = std::clamp(joystick->rx(),
                                cfg["ang_vel_z"][0].as<float>(),
                                cfg["ang_vel_z"][1].as<float>());
        }

        // write row
        g_csv << now_s() << "," << policy_step << "," << (est_updated ? 1 : 0);
        float pgx = data.projected_gravity_b[0];
        float pgy = data.projected_gravity_b[1];
        float pgz = data.projected_gravity_b[2];

        for (int i = 0; i < 12; ++i) g_csv << "," << (g_mu_valid ? g_mu_hat[i] : 0.0f);
        for (int i = 0; i < 12; ++i) g_csv << "," << (g_mu_raw_valid ? g_mu_raw[i] : 0.0f);

        g_csv << "," << pgx << "," << pgy << "," << pgz;
        g_csv << "," << wx << "," << wy << "," << wz;
        g_csv << "," << cmd_vx << "," << cmd_vy << "," << cmd_wz;

        for (int i = 0; i < 12; ++i) g_csv << "," << action_now[i];
        g_csv << "\n";

        // 避免每一行都 flush 太慢：你可以每 N 行 flush 一次
        if ((policy_step % 200) == 0) g_csv.flush();  // 約每 2 秒 flush (100Hz)
    }
} // namespace
State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");
    // ================== ⭐ 新增：載入 friction_estimator.onnx ==================
    try {
        auto estimator_path = (policy_dir / "exported" / "friction_estimator.onnx").string();
        spdlog::info("Loading friction estimator from {}", estimator_path);
        g_estimator = std::make_unique<FrictionEstimator>(estimator_path);
    } catch (const std::exception& e) {
        spdlog::error("Failed to create FrictionEstimator: {}", e.what());
        g_estimator.reset();
    }
    // ========================================================================

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    // 只初始化一次：你也可以把檔名加上 timestamp
    init_csv_once("/home/jeff/mujoco/unitree_rl_lab/deploy_logs/go2_run_log.csv");

    // 1) estimator（回傳這步有沒有真的推論）
    bool est_updated = update_friction_estimator(env.get());

    // 2) 拿 action
    auto action = env->action_manager->processed_actions();
    std::array<float, 12> action_now{};
    for (int i = 0; i < 12; ++i) action_now[i] = action[i];

    // 3) log 一行（這行會把 mu_hat + ang vel + cmd + action 寫進 CSV）
    log_csv_row(env.get(), g_policy_step, est_updated, action_now);

    // 4) 更新 last action（給下一步 estimator feature 用）
    for (int i = 0; i < 12; ++i) g_last_action[i] = action_now[i];

    // 5) 下到馬達
    for(int i = 0; i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action_now[i];
    }
}
