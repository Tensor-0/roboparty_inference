// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "robot_interface.hpp"

#include <array>
#include <cstdio>

RobotInterface::RobotInterface(const std::string& config_file) {
    YAML::Node config = YAML::LoadFile(config_file);

    imu_cfg_ = std::make_shared<IMUCfg>();
    if (config["imu"]) {
        YAML::Node imu_node = config["imu"];
        if (imu_node["imu_id"]) imu_cfg_->imu_id_ = imu_node["imu_id"].as<int>();
        if (imu_node["baudrate"]) imu_cfg_->baudrate_ = imu_node["baudrate"].as<int>();
        if (imu_node["imu_type"]) imu_cfg_->imu_type_ = imu_node["imu_type"].as<std::string>();
        if (imu_node["imu_interface_type"]) imu_cfg_->imu_interface_type_ = imu_node["imu_interface_type"].as<std::string>();
        if (imu_node["imu_interface"]) imu_cfg_->imu_interface_ = imu_node["imu_interface"].as<std::string>();
        setup_imu();
    }

    motors_cfg_ = std::make_shared<MotorsCfg>();
    if (config["motors"]) {
        YAML::Node motors_node = config["motors"];
        if (motors_node["motor_zero_offset"]) motors_cfg_->motor_zero_offset_ = motors_node["motor_zero_offset"].as<std::vector<double>>();
        if (motors_node["master_id_offset"]) motors_cfg_->master_id_offset_ = motors_node["master_id_offset"].as<int>();
        if (motors_node["motor_type"]) motors_cfg_->motor_type_ = motors_node["motor_type"].as<std::vector<std::string>>();
        if (motors_node["motor_interface_type"]) motors_cfg_->motor_interface_type_ = motors_node["motor_interface_type"].as<std::vector<std::string>>();
        if (motors_node["motor_interface"]) motors_cfg_->motor_interface_ = motors_node["motor_interface"].as<std::vector<std::string>>();
        if (motors_node["motor_id"]) motors_cfg_->motor_id_ = motors_node["motor_id"].as<std::vector<long int>>();
        if (motors_node["motor_model"]) motors_cfg_->motor_model_ = motors_node["motor_model"].as<std::vector<long int>>();
        if (motors_node["motor_num"]) motors_cfg_->motor_num_ = motors_node["motor_num"].as<std::vector<long int>>();
        // ⭐ 电机离线降级（2026-09-21）：阈值可配，默认 25（≈100ms @ dt=4ms）。
        //   ⚠️ 台架测试时设成 1 就能触发降级路径，不用真去拔 CAN 线
        //      （带电拔线会触发 gs_usb TX 假死，见 memory gs-usb-tx-wedge）。
        //   ⚠️ 别设 0 —— 那表示"只要有一次发送还没等到回复就算离线"，
        //      正常电机偶发一次回复晚到就会误触发。1 表示容忍 1 次。
        if (motors_node["offline_threshold"]) {
            offline_threshold_ = motors_node["offline_threshold"].as<int>();
            if (offline_threshold_ < 0) {
                offline_threshold_ = 0;
            }
        }
        setup_motors();
    } else {
        throw std::runtime_error("Motors configuration not found in " + config_file);
    }

    robot_cfg_ = std::make_shared<RobotCfg>();
    if (config["robot"]) {
        YAML::Node robot_node = config["robot"];
        if (robot_node["kp"]) robot_cfg_->kp_ = robot_node["kp"].as<std::vector<double>>();
        if (robot_node["kd"]) robot_cfg_->kd_ = robot_node["kd"].as<std::vector<double>>();
        if (robot_node["close_chain_motor_idx"]) robot_cfg_->close_chain_motor_idx_ = robot_node["close_chain_motor_idx"].as<std::vector<long int>>();
        if (robot_node["motor_sign"]) robot_cfg_->motor_sign_ = robot_node["motor_sign"].as<std::vector<long int>>();
        if (robot_node["urdf2motor"]) robot_cfg_->urdf2motor_ = robot_node["urdf2motor"].as<std::vector<long int>>();
        motor2urdf_ = std::vector<int>(motors_cfg_->motor_id_.size(), -1);
        for (size_t i = 0; i < robot_cfg_->urdf2motor_.size(); ++i) {
            motor2urdf_[robot_cfg_->urdf2motor_[i]] = i;
        }
        if (robot_node["extrinsic_R"]) {
            robot_cfg_->extrinsic_R_ = robot_node["extrinsic_R"].as<std::vector<double>>();
            if (robot_cfg_->extrinsic_R_.size() == 9) {
                // Row-major: [r00, r01, r02, r10, r11, r12, r20, r21, r22]
                extrinsic_R_mat_ << robot_cfg_->extrinsic_R_[0], robot_cfg_->extrinsic_R_[1], robot_cfg_->extrinsic_R_[2],
                                    robot_cfg_->extrinsic_R_[3], robot_cfg_->extrinsic_R_[4], robot_cfg_->extrinsic_R_[5],
                                    robot_cfg_->extrinsic_R_[6], robot_cfg_->extrinsic_R_[7], robot_cfg_->extrinsic_R_[8];
                Eigen::Quaternionf q_R(extrinsic_R_mat_);  // quaternion of R (Body->IMU)
                extrinsic_q_inv_ = q_R.inverse();           // we need R_inv for quaternion transform
            }
        }
        for (size_t ankle_idx = 0; ankle_idx < robot_cfg_->close_chain_motor_idx_.size(); ++ankle_idx) {
            const long int idx = robot_cfg_->close_chain_motor_idx_[ankle_idx];
            auto it = std::find(robot_cfg_->urdf2motor_.begin(), robot_cfg_->urdf2motor_.end(), idx);
            if (it != robot_cfg_->urdf2motor_.end()) {
                close_chain_joint_idx_.push_back(std::distance(robot_cfg_->urdf2motor_.begin(), it));
            }
        }
        if (robot_node["type"]) {
            ankle_decouple_ = Decouple::create(robot_node["type"].as<std::string>());
        } else {
            ankle_decouple_ = nullptr;
        }
    } else {
        throw std::runtime_error("Robot configuration not found in " + config_file);
    }

    thread_pool_ = std::make_unique<ThreadPool>(motors_cfg_->motor_interface_.size(), 46);

    joint_q_ = std::vector<float>(motors_cfg_->motor_id_.size(), 0.0);
    joint_vel_ = std::vector<float>(motors_cfg_->motor_id_.size(), 0.0);
    joint_tau_ = std::vector<float>(motors_cfg_->motor_id_.size(), 0.0);
    motor_pos_target_ = std::vector<float>(motors_cfg_->motor_id_.size(), 0.0);
    motor_vel_target_ = std::vector<float>(motors_cfg_->motor_id_.size(), 0.0);
    motor_kp_target_  = std::vector<float>(motors_cfg_->motor_id_.size(), 0.0);
    motor_kd_target_  = std::vector<float>(motors_cfg_->motor_id_.size(), 0.0);
    motor_tau_target_ = std::vector<float>(motors_cfg_->motor_id_.size(), 0.0);
}

void RobotInterface::setup_motors(){
    const size_t bus_count = motors_cfg_->motor_interface_.size();
    motor_bus_offsets_.assign(bus_count + 1, 0);
    for (size_t bus = 0; bus < bus_count; ++bus) {
        motor_bus_offsets_[bus + 1] = motor_bus_offsets_[bus] +
                                      static_cast<size_t>(motors_cfg_->motor_num_[bus]);
    }

    size_t count = 0;
    motors_.resize(motors_cfg_->motor_id_.size());
    for (size_t i = 0; i < motors_cfg_->motor_interface_.size(); ++i){
        for (size_t j = 0; j < motors_cfg_->motor_num_[i]; ++j){
            motors_[count] = MotorDriver::create_motor(motors_cfg_->motor_id_[count], motors_cfg_->motor_interface_type_[i], motors_cfg_->motor_interface_[i], motors_cfg_->motor_type_[i], motors_cfg_->motor_model_[count], motors_cfg_->master_id_offset_, motors_cfg_->motor_zero_offset_[count]);
            count += 1;
        }
    }
}

void RobotInterface::setup_imu(){
    imu_ = IMUDriver::create_imu(imu_cfg_->imu_id_, imu_cfg_->imu_interface_type_, imu_cfg_->imu_interface_, imu_cfg_->imu_type_, imu_cfg_->baudrate_);
}

void RobotInterface::forward_close_chain() {
    Eigen::Vector2d q, vel, tau;
    for (size_t pair = 0; pair < 2; ++pair) {
        const bool left = (pair == 0);
        int idx1 = close_chain_joint_idx_[pair * 2];
        int idx2 = close_chain_joint_idx_[pair * 2 + 1];
        q << joint_q_[idx1], joint_q_[idx2];
        vel << joint_vel_[idx1], joint_vel_[idx2];
        tau << joint_tau_[idx1], joint_tau_[idx2];
        ankle_decouple_->get_forwardQVT(q, vel, tau, left);
        joint_q_[idx1]   = q[0];
        joint_q_[idx2]   = q[1];
        joint_vel_[idx1] = vel[0];
        joint_vel_[idx2] = vel[1];
        joint_tau_[idx1] = tau[0];
        joint_tau_[idx2] = tau[1];
    }
}

void RobotInterface::read_joints(bool strict) {
    // ⭐ 电机离线降级（2026-09-21）：strict=false 时整条路都不抛。
    //   is_init_ 这里也要拦 —— 它是个 TOCTOU 竞态：按 X 失能的瞬间，
    //   apply_action 顶部的 is_init_ 检查可能已经通过，走到这里才翻。
    //   原来会抛 ⇒ control() 的外层 catch ⇒ rclcpp::shutdown() ⇒ 整个进程被按键干掉。
    if (!is_init_.load()) {
        if (!strict) {
            return;
        }
        throw std::runtime_error("Motors are not initialized");
    }
    std::unique_lock<std::mutex> lock(joint_mutex_);
    exec_motors_parallel([this](std::shared_ptr<MotorDriver>& motor, int idx) {
        joint_q_[motor2urdf_[idx]] = motor->get_motor_pos() * robot_cfg_->motor_sign_[idx];
        joint_vel_[motor2urdf_[idx]] = motor->get_motor_spd() * robot_cfg_->motor_sign_[idx];
        joint_tau_[motor2urdf_[idx]] = motor->get_motor_current() * robot_cfg_->motor_sign_[idx];
    });
    // ⚠️ 注意：离线时上面读到的是【上一帧缓存】的 joint_q_/joint_vel_（motor->get_motor_pos()
    //    返回的是缓存值，不发起新请求）⇒ 观测会【冻住】。这正是必须切 PD 的理由，
    //    不只是"电机坏了"——策略吃到不变的关节角会给出无意义的动作。
    if (strict) {
        throw_if_motors_offline();
    }

    if (!close_chain_joint_idx_.empty() && ankle_decouple_) {
        forward_close_chain();
    }
}

void RobotInterface::read_imu() {
    if (!imu_) {
        throw std::runtime_error("IMU is not initialized");
    }

    std::unique_lock<std::mutex> lock(imu_mutex_);
    const auto raw_quat = imu_->get_quat();          // w, x, y, z
    const auto raw_ang_vel = imu_->get_ang_vel();  // in IMU frame
    const auto raw_lin_acc = imu_->get_lin_acc();  // in IMU frame, m/s^2, gravity included
    Eigen::Quaternionf q_body =
        Eigen::Quaternionf(raw_quat[0], raw_quat[1], raw_quat[2], raw_quat[3]) * extrinsic_q_inv_;
    q_body.normalize();
    Eigen::Map<const Eigen::Vector3f> omega_imu(raw_ang_vel.data());
    const Eigen::Vector3f omega_body = extrinsic_R_mat_ * omega_imu;

    quat_buf_[0] = q_body.w();
    quat_buf_[1] = q_body.x();
    quat_buf_[2] = q_body.y();
    quat_buf_[3] = q_body.z();
    Eigen::Map<Eigen::Vector3f>(ang_vel_buf_.data()) = omega_body;
    Eigen::Map<const Eigen::Vector3f> acc_imu(raw_lin_acc.data());
    Eigen::Map<Eigen::Vector3f>(lin_acc_buf_.data()) = extrinsic_R_mat_ * acc_imu;
}

void RobotInterface::apply_action(const std::vector<float>& p,
                                  const std::vector<float>& v,
                                  const std::vector<float>& kp,
                                  const std::vector<float>& kd,
                                  const std::vector<float>& tau) {
    std::unique_lock<std::mutex> command_lock(command_mutex_);
    if(!is_init_.load()){
        return;
    }
    const bool use_close_chain_tau = !close_chain_joint_idx_.empty() && ankle_decouple_;
    std::array<float, 4> ankle_motor_tau{};

    // ⭐ 电机离线降级（2026-09-21）：strict=false。
    //   ⚠️ 这一行是本改动的核心 —— 少了它，read_joints 走默认 true 照旧抛异常，
    //      control() 的外层 catch 会 rclcpp::shutdown()，机器人瘫倒。
    //      台架实测抓到的就是这个：签名加了、声明加了、唯独这里没改（2026-09-21 17:53）。
    //   离线判定改由调用方 InferenceNode::apply_action() 用 motors_offline() 做。
    read_joints(false);

    {
        std::unique_lock<std::mutex> lock(joint_mutex_);
        if (use_close_chain_tau){
            auto kp_cc = [&](size_t i) -> double {
                return kp.empty() ? robot_cfg_->kp_[robot_cfg_->close_chain_motor_idx_[i]]
                                  : static_cast<double>(kp[close_chain_joint_idx_[i]]);
            };
            auto kd_cc = [&](size_t i) -> double {
                return kd.empty() ? robot_cfg_->kd_[robot_cfg_->close_chain_motor_idx_[i]]
                                  : static_cast<double>(kd[close_chain_joint_idx_[i]]);
            };
            auto vel_target_cc = [&](size_t i) -> double {
                return v.empty() ? 0.0 : static_cast<double>(v[close_chain_joint_idx_[i]]);
            };
            auto tau_ff_cc = [&](size_t i) -> double {
                return tau.empty() ? 0.0 : static_cast<double>(tau[close_chain_joint_idx_[i]]);
            };

            Eigen::Vector2d q, vel, tau_cc;
            for (size_t pair = 0; pair < 2; ++pair) {
                const bool left = (pair == 0);
                const size_t off = pair * 2;
                const int idx1 = close_chain_joint_idx_[off];
                const int idx2 = close_chain_joint_idx_[off + 1];
                q << joint_q_[idx1], joint_q_[idx2];
                vel << joint_vel_[idx1], joint_vel_[idx2];
                tau_cc << kp_cc(off)     * (p[idx1] - q[0]) + kd_cc(off)     * (vel_target_cc(off)     - vel[0]) + tau_ff_cc(off),
                          kp_cc(off + 1) * (p[idx2] - q[1]) + kd_cc(off + 1) * (vel_target_cc(off + 1) - vel[1]) + tau_ff_cc(off + 1);
                ankle_decouple_->get_decoupleQVT(q, vel, tau_cc, left);
                ankle_motor_tau[off] = static_cast<float>(tau_cc[0]);
                ankle_motor_tau[off + 1] = static_cast<float>(tau_cc[1]);
            }
        }
    }

    {
        std::unique_lock<std::mutex> lock(motors_mutex_);
        for (size_t i = 0; i < motor_pos_target_.size(); i++){
            const size_t ji = motor2urdf_[i];
            motor_pos_target_[i] = p[ji];
            motor_vel_target_[i] = v.empty() ? 0.0f : v[ji];
            motor_kp_target_[i]  = kp.empty() ? static_cast<float>(robot_cfg_->kp_[i]) : kp[ji];
            motor_kd_target_[i]  = kd.empty() ? static_cast<float>(robot_cfg_->kd_[i]) : kd[ji];
            motor_tau_target_[i] = tau.empty() ? 0.0f : tau[ji];
        }
        if (use_close_chain_tau) {
            for (size_t ankle_idx = 0; ankle_idx < ankle_motor_tau.size(); ++ankle_idx) {
                const size_t motor_idx = robot_cfg_->close_chain_motor_idx_[ankle_idx];
                motor_pos_target_[motor_idx] = 0.0f;
                motor_vel_target_[motor_idx] = 0.0f;
                motor_kp_target_[motor_idx] = 0.0f;
                motor_kd_target_[motor_idx] = 0.0f;
                motor_tau_target_[motor_idx] = ankle_motor_tau[ankle_idx];
            }
        }
    }

    motors_mit_cmd();
}

void RobotInterface::reset_joints(std::vector<double> joint_default_angle) {
    std::unique_lock<std::mutex> reset_lock(reset_mutex_, std::try_to_lock);
    if (!reset_lock.owns_lock()) {
        throw std::runtime_error("Joint reset is already in progress");
    }
    std::unique_lock<std::mutex> command_lock(command_mutex_);

    constexpr int reset_duration_ms = 4000;
    constexpr int reset_period_ms = 20;
    constexpr int reset_steps = reset_duration_ms / reset_period_ms;
    const auto reset_period = std::chrono::milliseconds(reset_period_ms);

    refresh_joints();
    const std::vector<float> start_joint_q = get_joint_q();
    auto next_frame = std::chrono::steady_clock::now();

    for (int step = 0; step <= reset_steps; ++step) {
        if (step > 0) {
            read_joints();
        }

        const double phase = static_cast<double>(step) / reset_steps;
        const double blend = phase * phase * (3.0 - 2.0 * phase);
        std::vector<double> joint_target(joint_default_angle.size());
        for (size_t i = 0; i < joint_target.size(); ++i) {
            joint_target[i] = start_joint_q[i] +
                              blend * (joint_default_angle[i] - start_joint_q[i]);
        }

        if (!close_chain_joint_idx_.empty() && ankle_decouple_) {
            Eigen::Vector2d q, vel = Eigen::Vector2d::Zero(), tau = Eigen::Vector2d::Zero();
            for (size_t pair = 0; pair < 2; ++pair) {
                const bool left = (pair == 0);
                int idx1 = close_chain_joint_idx_[pair * 2];
                int idx2 = close_chain_joint_idx_[pair * 2 + 1];
                q << joint_target[idx1], joint_target[idx2];
                ankle_decouple_->get_decoupleQVT(q, vel, tau, left);
                joint_target[idx1] = q[0];
                joint_target[idx2] = q[1];
            }
        }

        {
            std::unique_lock<std::mutex> lock(motors_mutex_);
            for (size_t i = 0; i < motor_pos_target_.size(); i++){
                motor_pos_target_[i] = static_cast<float>(joint_target[motor2urdf_[i]]);
                motor_vel_target_[i] = 0.0f;
                motor_kp_target_[i]  = static_cast<float>(robot_cfg_->kp_[i]) * (1.0f / 2.5f);
                motor_kd_target_[i]  = static_cast<float>(robot_cfg_->kd_[i]);
                motor_tau_target_[i] = 0.0f;
            }
        }

        motors_mit_cmd();
        if (step < reset_steps) {
            next_frame += reset_period;
            std::this_thread::sleep_until(next_frame);
        }
    }

    {
        std::unique_lock<std::mutex> lock(motors_mutex_);
        for (size_t i = 0; i < motor_kp_target_.size(); i++){
            motor_kp_target_[i] = static_cast<float>(robot_cfg_->kp_[i]);
        }
    }
    motors_mit_cmd();
}

void RobotInterface::refresh_joints() {
    if (!is_init_.load()) {
        throw std::runtime_error("Motors are not initialized");
    }
    exec_motors_parallel([](std::shared_ptr<MotorDriver>& motor, int) {
        motor->refresh_motor_status();
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    read_joints();
}

void RobotInterface::set_zeros() {
    std::unique_lock<std::mutex> command_lock(command_mutex_);
    if (!is_init_.load()) {
        throw std::runtime_error("Motors are not initialized");
    }
    exec_motors_parallel([](std::shared_ptr<MotorDriver>& motor, int) {
        motor->set_motor_zero();
    });
}

void RobotInterface::clear_errors() {
    std::unique_lock<std::mutex> command_lock(command_mutex_);
    exec_motors_parallel([](std::shared_ptr<MotorDriver>& motor, int) {
        motor->clear_motor_error();
    });
}

// ⭐ init 结果消费（2026-09-22）
//
//   原实现把 init_motor() 的返回值【丢掉】，然后无条件 is_init_ = true
//   ⇒ 有一个电机没使能成功也算"初始化成功"，操作员毫无察觉地去站立。
//
//   两层判据，缺一不可：
//
//     ① 返回值。四个驱动（DM / XYN / LRO / EVO）语义一致：0 = 正常，非 0 = 故障码。
//        ⚠️ 但 0 只说明【没报故障】，不说明"使能成功" —— DM 的 error_id_ 只在
//           反馈帧高 4 位 > 7 时才写（dm_motor_driver.cpp:238），而 err = 1（使能）
//           本来就 ≤ 7 ⇒ 它永远是初值 0，判不了"使能"这件事。
//           （memory: dm-driver-error-id-bug，这个坑踩过）
//
//     ② 存活。每台【单独】refresh，等 60 ms 看有没有回复，最多 3 轮。
//        和运行时离线判定是同一个原语：每收到一帧就清零，且【先按电机 ID 过滤】
//        （dm_motor_driver.cpp:224-229），别的电机的回复不算数。
//        ⚠️ 这里不能用 offline_threshold_（25）当阈值：init 才发了几帧，
//           拿 25 当阈值等于没检查。
//        ⚠️ 也不能"一次把多台都刷了再看" —— 见下面 kProbeAttempts 处的实测记录。
//        ⚠️ CANFD 的一条 MIT 帧由总线上的 0 号电机代发，所以运行时那个
//           offline_threshold_ 判定对同总线其它电机是失效的；但 refresh 是
//           【每台电机各发一帧】的，所以这里的探针对 CAN 和 CANFD 都成立。
//
//   任一电机没过 ⇒ 全体失能（fail-closed）并把 is_init_ 留在 false。
//   为什么整体拒绝而不是只警告：9/10 个电机使能的机器人站不住，而操作员
//   未必看得出来；失能的那个可能正是承重腿。宁可上不了电，不要站到一半垮掉。
void RobotInterface::init_motors() {
    std::unique_lock<std::mutex> command_lock(command_mutex_);
    if (is_init_.load()) {
        throw std::runtime_error("Motors are already initialized");
    }

    const auto describe_motor = [this](size_t idx) {
        const int joint_idx = (idx < motor2urdf_.size()) ? motor2urdf_[idx] : -1;
        return motors_[idx]->get_can_name() + "/id=" +
               std::to_string(motors_cfg_->motor_id_[idx]) + " (joint " +
               std::to_string(joint_idx) + ")";
    };

    std::vector<std::string> failures;
    std::mutex failures_mutex;
    exec_motors_parallel([&](std::shared_ptr<MotorDriver>& motor, int idx) {
        std::string failure;
        try {
            const uint8_t code = motor->init_motor();
            if (code != 0) {
                char code_text[8];
                std::snprintf(code_text, sizeof(code_text), "0x%02X", code);
                failure = std::string("init_motor() returned ") + code_text + " (fault)";
            }
        } catch (const std::exception& e) {
            // 让每台电机都把话说完：一条总线上有电机抛异常，不该挡住别的电机
            // 报出自己的问题，也不该让我们在半初始化状态下直接往上层抛。
            failure = std::string("init_motor() threw: ") + e.what();
        }
        if (!failure.empty()) {
            std::lock_guard<std::mutex> lock(failures_mutex);
            failures.push_back(describe_motor(static_cast<size_t>(idx)) + ": " + failure);
        }
    });

    if (failures.empty()) {
        // ⭐ 存活探针（2026-09-22，第二版：逐台 + 短窗口 + 重试）
        //
        //   为什么不是"一次把 5 台都刷了、等 50 ms 再看"：
        //   2026-09-22 真机上那版【反复假阳性】—— 10 台里总有两台报没回复，
        //   而且每次换一对、还换一条总线（先是 can1 的 id=3/4，再是 can0 的 id=3/4）。
        //   同一时刻 CAN 层是干净的（ERROR-ACTIVE、错误计数全 0），
        //   慢速逐台探（每台单独刷、间隔 60 ms）则 10/10 稳定有回复。
        //   ⇒ 是探法太急，不是电机坏了。假阳性的代价是"机器人根本没法上电"。
        //
        //   现在按实测可靠的那个节奏来：每台单独刷、等 60 ms、最多 3 轮。
        //   真死的电机三轮都不会回 ⇒ 判据没松，只是不拿单次慢回当事。
        //   开销：每总线 5 台 × 60 ms ≈ 300 ms（两条总线并行）。
        constexpr int kProbeAttempts = 3;
        constexpr int kProbeWindowMs = 60;
        // ⚠️ 用 vector<char> 而不是 vector<bool>：后者是【位压缩】的，
        //    两条总线各跑一个线程、往不同下标写，会落在同一个字上 ⇒ 丢更新
        //    ⇒ 明明有回复的电机被记成没回复（实测 10 台里随机挂 4 台，
        //    而单台实验和逐台脚本探针都是 10/10 正常）。
        std::vector<char> answered(motors_.size(), 0);
        exec_motors_parallel([&](std::shared_ptr<MotorDriver>& motor, int idx) {
            const size_t i = static_cast<size_t>(idx);
            for (int attempt = 0; attempt < kProbeAttempts && !answered[i]; ++attempt) {
                motor->refresh_motor_status();
                std::this_thread::sleep_for(std::chrono::milliseconds(kProbeWindowMs));
                if (motor->get_response_count() == 0) {
                    answered[i] = 1;
                }
            }
        });
        for (size_t idx = 0; idx < motors_.size(); ++idx) {
            if (!answered[idx]) {
                std::lock_guard<std::mutex> lock(failures_mutex);
                failures.push_back(describe_motor(idx) + ": no reply after " +
                                   std::to_string(kProbeAttempts) + " refreshes");
            }
        }
    }

    if (!failures.empty()) {
        std::sort(failures.begin(), failures.end());
        std::string message = "init_motors failed (" + std::to_string(failures.size()) +
                              "/" + std::to_string(motors_.size()) + "): ";
        for (size_t i = 0; i < failures.size(); ++i) {
            if (i > 0) {
                message += "; ";
            }
            message += failures[i];
        }
        message += " -- all motors deinitialized, is_init_ stays false";
        exec_motors_parallel([](std::shared_ptr<MotorDriver>& motor, int) {
            motor->deinit_motor();
        });
        throw std::runtime_error(message);
    }

    is_init_.store(true);
}

void RobotInterface::deinit_motors() {
    if (!is_init_.exchange(false)) {
        throw std::runtime_error("Motors are already deinitialized");
    }
    exec_motors_parallel([](std::shared_ptr<MotorDriver>& motor, int) {
        motor->deinit_motor();
    });
}

void RobotInterface::motors_mit_cmd() {
    std::unique_lock<std::mutex> lock(motors_mutex_);
    if (!is_init_.load()) {
        return;
    }
    thread_pool_->run_parallel(motor_bus_offsets_.size() - 1, [this](size_t bus) {
        const size_t start_count = motor_bus_offsets_[bus];
        const size_t end_count = motor_bus_offsets_[bus + 1];
        if (motors_cfg_->motor_interface_type_[bus] == "canfd") {
            float pos[8] = {}, vel[8] = {}, kp[8] = {}, kd[8] = {}, tau[8] = {};
            for (size_t idx = start_count; idx < end_count; ++idx) {
                const long int motor_id = motors_cfg_->motor_id_[idx];
                const size_t slot = (motor_id > 0 && motor_id <= 8)
                    ? static_cast<size_t>(motor_id - 1)
                    : idx - start_count;
                if (slot >= 8) continue;
                const float sign = static_cast<float>(robot_cfg_->motor_sign_[idx]);
                pos[slot] = motor_pos_target_[idx] * sign;
                vel[slot] = motor_vel_target_[idx] * sign;
                kp[slot]  = motor_kp_target_[idx];
                kd[slot]  = motor_kd_target_[idx];
                tau[slot] = motor_tau_target_[idx] * sign;
            }
            motors_[start_count]->motor_mit_cmd(pos, vel, kp, kd, tau);
        } else {
            for (size_t idx = start_count; idx < end_count; ++idx) {
                const float sign = static_cast<float>(robot_cfg_->motor_sign_[idx]);
                motors_[idx]->motor_mit_cmd(motor_pos_target_[idx] * sign,
                                             motor_vel_target_[idx] * sign,
                                             motor_kp_target_[idx],
                                             motor_kd_target_[idx],
                                             motor_tau_target_[idx] * sign);
            }
        }
    });
}

// ⭐ 电机离线降级（2026-09-21）：拆成"查询"和"抛"两件事。
//   motors_offline()   —— 热路径用（control 线程 250Hz），短路返回，不建字符串、不加锁
//   offline_motor_list() —— 日志用，只在真的要报的时候才建字符串
//   throw_if_motors_offline() —— 服务/标定路径用，语义与改动前完全一致
bool RobotInterface::motors_offline() const {
    for (size_t idx = 0; idx < motors_.size(); ++idx) {
        if (motors_[idx]->get_response_count() > offline_threshold_) {
            return true;
        }
    }
    return false;
}

std::string RobotInterface::offline_motor_list() const {
    std::string offline_motors;
    for (size_t idx = 0; idx < motors_.size(); ++idx) {
        const int response_count = motors_[idx]->get_response_count();
        if (response_count <= offline_threshold_) {
            continue;
        }
        if (!offline_motors.empty()) {
            offline_motors += ", ";
        }
        offline_motors += motors_[idx]->get_can_name() + "/id=" +
                          std::to_string(motors_cfg_->motor_id_[idx]) +
                          "(count=" + std::to_string(response_count) + ")";
    }
    return offline_motors;
}

// ⭐ 温度监控（2026-09-22）：见头文件里的说明。只读 atomic，不加锁。
float RobotInterface::max_motor_temperature() const {
    float hottest = 0.0f;
    for (const auto& motor : motors_) {
        const float temperature = motor->get_motor_temperature();
        if (temperature > hottest) {
            hottest = temperature;
        }
    }
    return hottest;
}

std::string RobotInterface::hot_motor_list(float limit) const {
    std::string hot_motors;
    for (size_t idx = 0; idx < motors_.size(); ++idx) {
        const float temperature = motors_[idx]->get_motor_temperature();
        if (temperature <= limit) {
            continue;
        }
        if (!hot_motors.empty()) {
            hot_motors += ", ";
        }
        hot_motors += motors_[idx]->get_can_name() + "/id=" +
                      std::to_string(motors_cfg_->motor_id_[idx]) + "(" +
                      std::to_string(static_cast<long>(std::lround(temperature))) + "C)";
    }
    return hot_motors.empty() ? "none" : hot_motors;
}

// ⭐ IMU 数据新鲜度（2026-09-22）：见 imu_driver.hpp 里的实测记录。
//   ⚠️ 这里必须用 steady_clock，不能和驱动那边的时钟混用别的源。
float RobotInterface::imu_data_age_s() const {
    if (!imu_) {
        return -1.0f;
    }
    const int64_t last_ns = imu_->last_frame_ns();
    if (last_ns == 0) {
        return -1.0f;  // 一帧都没收到过 —— 不是"陈旧"，交给 NaN 守卫
    }
    const int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now().time_since_epoch())
                               .count();
    return static_cast<float>(static_cast<double>(now_ns - last_ns) * 1e-9);
}

void RobotInterface::throw_if_motors_offline() const {
    const std::string offline_motors = offline_motor_list();
    if (!offline_motors.empty()) {
        throw std::runtime_error("Motors offline: " + offline_motors);
    }
}
