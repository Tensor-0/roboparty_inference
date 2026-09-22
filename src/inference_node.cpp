// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "inference_node.hpp"

#include <fstream>  // A3：算配置文件指纹
#include <limits>   // quiet_NaN —— 注入用

void InferenceNode::update_obs_history(std::vector<float>& history,
                                       const std::vector<float>& obs,
                                       int obs_num, int frame_stack,
                                       bool is_first_frame) {
    if (is_first_frame) {
        for (int frame = 0; frame < frame_stack; frame++) {
            std::copy(obs.begin(), obs.end(), history.begin() + frame * obs_num);
        }
        return;
    }
    std::move(history.begin() + obs_num,
              history.begin() + frame_stack * obs_num,
              history.begin());
    std::copy(obs.begin(), obs.end(), history.begin() + (frame_stack - 1) * obs_num);
}

ObsStackOrder InferenceNode::parse_obs_stack_order(const std::string& stack_order_name) {
    if (stack_order_name == "frame_major") {
        return ObsStackOrder::FrameMajor;
    }
    if (stack_order_name == "obs_major") {
        return ObsStackOrder::ObsMajor;
    }
    throw std::runtime_error("Unsupported obs stack order: " + stack_order_name);
}

void InferenceNode::update_stacked_obs(std::vector<float>& input_buffer, const std::vector<float>& obs,
                                       int obs_num, int frame_stack, ObsStackOrder stack_order,
                                       const std::vector<int>& field_sizes, bool is_first_frame) {
    if (stack_order == ObsStackOrder::FrameMajor) {
        update_obs_history(input_buffer, obs, obs_num, frame_stack, is_first_frame);
        return;
    }

    int input_offset = 0;
    int obs_offset = 0;

    for (const int field_size : field_sizes) {
        if (is_first_frame) {
            for (int frame = 0; frame < frame_stack; frame++) {
                std::copy(obs.begin() + obs_offset, obs.begin() + obs_offset + field_size,
                          input_buffer.begin() + input_offset + frame * field_size);
            }
        } else {
            std::move(input_buffer.begin() + input_offset + field_size,
                      input_buffer.begin() + input_offset + frame_stack * field_size,
                      input_buffer.begin() + input_offset);
            std::copy(obs.begin() + obs_offset, obs.begin() + obs_offset + field_size,
                      input_buffer.begin() + input_offset + (frame_stack - 1) * field_size);
        }
        input_offset += field_size * frame_stack;
        obs_offset += field_size;
    }
}

void InferenceNode::gather_sparse_obs_history(
    std::vector<float>& input_buffer,
    const std::vector<float>& obs_history,
    const std::vector<ObsHistorySlice>& gather_plan) {
    auto output = input_buffer.begin();
    for (const ObsHistorySlice& slice : gather_plan) {
        output = std::copy_n(
            obs_history.begin() + slice.history_offset, slice.size, output);
    }
}

void InferenceNode::setup_model(std::unique_ptr<ModelContext>& ctx, std::string model_path, int input_size) {
    if (!ctx) {
        ctx = std::make_unique<ModelContext>();
    }

    Ort::SessionOptions session_options;
    session_options.DisablePerSessionThreads();
    session_options.EnableCpuMemArena();
    session_options.EnableMemPattern();
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    ctx->session = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);
    
    ctx->num_inputs = ctx->session->GetInputCount();
    if (ctx->num_inputs != 1) {
        throw std::runtime_error("Only single-input ONNX models are supported: " + model_path);
    }
    ctx->input_names.resize(ctx->num_inputs);

    for (size_t i = 0; i < ctx->num_inputs; i++) {
        Ort::AllocatedStringPtr input_name = ctx->session->GetInputNameAllocated(i, allocator_);
        ctx->input_names[i] = input_name.get();
        auto type_info = ctx->session->GetInputTypeInfo(i);
        ctx->input_shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
        if (ctx->input_shape[0] == -1) ctx->input_shape[0] = 1;
    }

    size_t model_input_size = 1;
    for (size_t i = 0; i < ctx->input_shape.size(); i++) {
        model_input_size *= static_cast<size_t>(ctx->input_shape[i]);
    }
    if (model_input_size != static_cast<size_t>(input_size)) {
        throw std::runtime_error(
            "ONNX input size mismatch for " + model_path + ": model expects " +
            std::to_string(model_input_size) + " values, but config provides " + std::to_string(input_size));
    }
    ctx->input_buffer.resize(input_size);

    ctx->num_outputs = ctx->session->GetOutputCount();
    ctx->output_names.resize(ctx->num_outputs);
    ctx->output_buffer.resize(joint_num_);

    for (size_t i = 0; i < ctx->num_outputs; i++) {
        Ort::AllocatedStringPtr output_name = ctx->session->GetOutputNameAllocated(i, allocator_);
        ctx->output_names[i] = output_name.get();
        auto type_info = ctx->session->GetOutputTypeInfo(i);
        ctx->output_shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
        if (ctx->output_shape[0] == -1) ctx->output_shape[0] = 1;
        if (ctx->output_shape[1] == -1) ctx->output_shape[1] = joint_num_;
    }

    ctx->input_names_raw = std::vector<const char *>(ctx->num_inputs, nullptr);
    ctx->output_names_raw = std::vector<const char *>(ctx->num_outputs, nullptr);
    for (size_t i = 0; i < ctx->num_inputs; i++) {
        ctx->input_names_raw[i] = ctx->input_names[i].c_str();
    }
    for (size_t i = 0; i < ctx->num_outputs; i++) {
        ctx->output_names_raw[i] = ctx->output_names[i].c_str();
    }

    ctx->memory_info = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));
    
    ctx->input_tensor = std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(
        *ctx->memory_info, ctx->input_buffer.data(), ctx->input_buffer.size(), ctx->input_shape.data(), ctx->input_shape.size()));
        
    ctx->output_tensor = std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(
        *ctx->memory_info, ctx->output_buffer.data(), ctx->output_buffer.size(), ctx->output_shape.data(), ctx->output_shape.size()));
}

// ⭐ PD 站立保底（2026-09-17）
//   switch_to_pd_stand：切到 PD 站立（目标由 control 线程按 pd_stand_target_ 写）。
//     ⚠️ 不直接改 last_act_ —— 让 control 线程里的 act_alpha 平滑把它带过去（防跳变）。
//   switch_to_policy：切回策略。
//   ⚠️ 切到 PD 后【不自动切回】（跌倒是"出事"，自动切回会在阈值附近反复横跳）。
//
// ⭐ A2（2026-09-22）：这两处是【唯一】的模式变更点，所以事件话题也挂在这里 ——
//   每次真变更发一条 /act_mode（JSON）。detail 用来带上下文（哪个电机离线、
//   漏了多少 ms），否则事后只有一句"切 PD 了"，等于没有信息。
//   ⚠️ 这里可能在 control 线程（实时）里被调用：拼 JSON 会分配内存，所以
//      只在【真的变了】那一刻做（上面的 exchange 保证），不是每个周期。
static std::string json_escape(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (const char c : text) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += c;
        } else if (c == '\n') {
            out += "\\n";
        } else {
            out += c;
        }
    }
    return out;
}

void InferenceNode::publish_act_mode(const char* prev_mode, const char* new_mode,
                                     const std::string& reason, const std::string& detail) {
    if (!act_mode_publisher_) {
        return;  // 构造过程中就会调用（初始化那一刻还没建发布者）
    }
    std::string payload = std::string("{\"stamp_ns\":") + std::to_string(this->now().nanoseconds()) +
                          ",\"prev\":\"" + json_escape(prev_mode) +
                          "\",\"mode\":\"" + json_escape(new_mode) +
                          "\",\"reason\":\"" + json_escape(reason) + "\"";
    if (!detail.empty()) {
        payload += ",\"detail\":\"" + json_escape(detail) + "\"";
    }
    payload += "}";
    std_msgs::msg::String msg;
    msg.data = payload;
    act_mode_publisher_->publish(msg);
}

// ⭐ A3（2026-09-22）：文件内容指纹。
//   ⚠️ 这是 FNV-1a 64 位，不是密码学哈希 —— 只用来回答"是不是同一个文件"，
//      不承担防篡改。节点里不引加密库（为这个引一个 OpenSSL 不划算）。
//   ⚠️ onnx 的【模型身份】不该用文件哈希：跨导出环境字节会变。
//      那是【权重指纹】的活（toolkit 的 _trace_policy_provenance.py：按名字排序
//      initializer 再哈希）。节点只负责把那个值从 sidecar 文件读出来原样报出。
static std::string fnv1a64_hex(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.good()) {
        return "MISSING";
    }
    uint64_t hash = 1469598103934665603ULL;
    char buf[4096];
    while (in.read(buf, sizeof(buf)) || in.gcount() > 0) {
        for (std::streamsize i = 0; i < in.gcount(); ++i) {
            hash ^= static_cast<unsigned char>(buf[i]);
            hash *= 1099511628211ULL;
        }
    }
    std::ostringstream out;
    out << std::hex << hash;
    return out.str();
}

static std::string read_first_line(const std::string& path) {
    std::ifstream in(path);
    if (!in.good()) {
        return "";
    }
    std::string line;
    std::getline(in, line);
    return line;
}

void InferenceNode::publish_node_metadata() {
    if (!node_metadata_publisher_) {
        return;
    }
    const auto quote = [](const std::string& text) { return "\"" + json_escape(text) + "\""; };
    const auto hex_or_missing = [](const std::string& path) {
        const std::string h = fnv1a64_hex(path);
        return h == "MISSING" ? std::string("\"MISSING\"") : "\"" + h + "\"";
    };
    const auto numbers = [](const std::vector<double>& values) {
        std::string out = "[";
        for (size_t i = 0; i < values.size(); ++i) {
            if (i > 0) out += ",";
            out += std::to_string(values[i]);
        }
        return out + "]";
    };
    const auto strings = [&quote](const std::vector<std::string>& values) {
        std::string out = "[";
        for (size_t i = 0; i < values.size(); ++i) {
            if (i > 0) out += ",";
            out += quote(values[i]);
        }
        return out + "]";
    };

    // sidecar：由【部署那一步】写的旁挂文件。没有就报 unknown —— 别假装知道。
    //
    // ⚠️ 必须顺着软链走到【真文件】再找 sidecar：--symlink-install 只给
    //    "构建时已存在"的文件建软链，而 GIT_SHA / 权重指纹是部署那一步才写进去的
    //    ⇒ install 树里根本没有它们的软链（实测报 unknown）。canonical 之后
    //    路径落在源码树里，那里才是部署脚本真正写文件的地方。
    std::error_code ec;
    const auto real_path = [&ec](const std::string& path) {
        const auto canonical = std::filesystem::canonical(path, ec);
        if (ec) {
            ec.clear();
            return std::filesystem::path(path);
        }
        return canonical;
    };
    const std::string model_path = policies_.empty() ? "" : policies_[0].model_path;
    const std::filesystem::path real_model = real_path(model_path);
    const std::filesystem::path real_config = real_path(robot_config_path_);
    const std::string fingerprint = model_path.empty()
        ? "" : read_first_line(real_model.string() + ".weight_fingerprint");
    const std::filesystem::path package_root =
        real_config.parent_path().parent_path().parent_path();
    const std::string git_sha = read_first_line((package_root / "GIT_SHA").string());

    // obs 布局从【解析后的 policy】渲染，不存原文 —— 报出来的就是真正在用的那个布局
    std::vector<std::string> layouts;
    std::vector<std::string> stacks;
    std::vector<std::string> orders;
    for (const PolicyRuntime& policy : policies_) {
        std::string layout;
        for (const ObsSourceSpec& spec : policy.obs_layout) {
            if (!layout.empty()) layout += ", ";
            layout += spec.name + ":" + std::to_string(spec.size);
        }
        layouts.push_back(layout);
        stacks.push_back(std::to_string(policy.frame_stack));
        orders.push_back(policy.stack_order == ObsStackOrder::FrameMajor ? "frame_major"
                                                                         : "obs_major");
    }

    std::vector<std::pair<std::string, std::string>> items = {
        {"stamp_ns", std::to_string(this->now().nanoseconds())},
        {"robot_name", quote(robot_name_)},
        {"policy_name", quote(policy_name_)},
        {"robot_config", quote(real_config.string())},
        {"robot_config_fnv1a64", hex_or_missing(real_config.string())},
        {"policy_config", quote(policy_config_path_.empty()
                                    ? "" : real_path(policy_config_path_).string())},
        {"policy_config_fnv1a64", policy_config_path_.empty()
                                      ? "\"MISSING\""
                                      : hex_or_missing(real_path(policy_config_path_).string())},
        {"onnx", quote(real_model.string())},
        // ⚠️ file_size 对不存在的文件会抛 —— 元数据不该把节点搞崩
        {"onnx_bytes", std::to_string(
             (!model_path.empty() && std::filesystem::exists(real_model))
                 ? static_cast<long long>(std::filesystem::file_size(real_model)) : 0)},
        {"onnx_weight_fingerprint", quote(fingerprint.empty() ? "unknown" : fingerprint)},
        {"git_sha", quote(git_sha.empty() ? "unknown" : git_sha)},
        // 生效参数（override 之后）—— 离线回放要的就是这些，文件里的值不算数
        {"joint_default_angle", numbers(joint_default_angle_)},
        {"pd_stand_target", numbers(pd_stand_target_)},
        {"joint_limits", numbers(joint_limits_)},
        {"action_scale", numbers(action_scale_)},
        {"clip_cmd", numbers(clip_cmd_)},
        {"obs_layouts", strings(layouts)},
        {"frame_stacks", strings(stacks)},
        {"obs_stack_orders", strings(orders)},
        {"usd2urdf", numbers(std::vector<double>(usd2urdf_.begin(), usd2urdf_.end()))},
        {"clip_actions", std::to_string(clip_actions_)},
        {"clip_observations", std::to_string(clip_observations_)},
        {"action_rescale", std::to_string(action_rescale_)},
        {"act_alpha", std::to_string(act_alpha_)},
        {"dt", std::to_string(dt_)},
        {"decimation", std::to_string(decimation_)},
        {"gravity_z_upper", std::to_string(gravity_z_upper_)},
        {"cmd_timeout_s", std::to_string(cmd_timeout_s_)},
        {"imu_timeout_s", std::to_string(imu_timeout_s_)},
        {"motor_temp_warn_c", std::to_string(temp_warn_c_)},
        {"motor_temp_cutoff_c", std::to_string(temp_cutoff_c_)},
        {"start_mode", quote(start_mode_policy_ ? "policy" : "pd_stand")},
        {"safety_inject_enabled", safety_inject_enabled_ ? "true" : "false"},
    };

    std::string payload = "{";
    for (size_t i = 0; i < items.size(); ++i) {
        if (i > 0) payload += ",";
        payload += quote(items[i].first) + ":" + items[i].second;
    }
    payload += "}";
    std_msgs::msg::String msg;
    msg.data = payload;
    node_metadata_publisher_->publish(msg);
    RCLCPP_INFO(this->get_logger(),
                "node metadata: git_sha=%s onnx_fingerprint=%s robot_config_fnv=%s",
                git_sha.empty() ? "unknown" : git_sha.c_str(),
                fingerprint.empty() ? "unknown" : fingerprint.c_str(),
                fnv1a64_hex(real_config.string()).c_str());
}

void InferenceNode::switch_to_pd_stand(const char* reason, const std::string& detail) {
    ActMode prev = act_mode_.exchange(ActMode::PD_STAND);
    if (prev != ActMode::PD_STAND) {
        RCLCPP_WARN(this->get_logger(),
                    "act_mode: POLICY -> PD_STAND  (%s)", reason ? reason : "");
        publish_act_mode("POLICY", "PD_STAND", reason ? reason : "", detail);
    }
}

void InferenceNode::switch_to_policy() {
    ActMode prev = act_mode_.exchange(ActMode::POLICY);
    if (prev != ActMode::POLICY) {
        RCLCPP_INFO(this->get_logger(), "act_mode: PD_STAND -> POLICY");
        publish_act_mode("PD_STAND", "POLICY", "manual switch", "");
        // 切回策略时让推理从干净状态开始（清 last_act_ 会让它突然弹回，
        // 所以只重置策略内部状态，动作仍由 act_alpha 平滑过渡）
        std::unique_lock<std::mutex> lock(act_mutex_);
        for (auto& policy : policies_) {
            policy.is_first_frame = true;
        }
    }
}

void InferenceNode::reset_runtime_state() {
    is_running_.store(false);
    std::unique_lock<std::mutex> mode_lock(mode_mutex_);
    std::unique_lock<std::mutex> control_lock(control_mutex_);
    is_interrupt_.store(false);
    is_motion_policy_.store(false);
    active_policy_idx_ = 0;
    {
        std::unique_lock<std::mutex> lock(cmd_mutex_);
        std::fill(cmd_vel_.begin(), cmd_vel_.end(), 0.0f);
    }
    {
        std::unique_lock<std::mutex> lock(perception_mutex_);
        std::fill(perception_obs_buffer_.begin(), perception_obs_buffer_.end(), 0.0f);
    }
    {
        std::unique_lock<std::mutex> lock(act_mutex_);
        // ⭐ PD 站立目标（2026-09-22）：复位时停在【保底姿态】而不是策略参考系。
        //   启动模式默认就是 pd_stand，所以这才是"上电后第一帧要摆成什么样"。
        //   （pd_stand_target_ 留空时 load_config 已把它填成 joint_default_angle_，
        //    所以这条改动对没配该项的机器人逐位等价。）
        for (int i = 0; i < joint_num_ && i < static_cast<int>(pd_stand_target_.size()); i++) {
            act_[i] = static_cast<float>(pd_stand_target_[i]);
            last_act_[i] = static_cast<float>(pd_stand_target_[i]);
        }
    }
    if (supports_interrupt()) {
        if (joint_default_angle_.size() < interrupt_action_.size()) {
            throw std::runtime_error("joint_default_angle is smaller than interrupt_action");
        }
        std::unique_lock<std::mutex> lock(interrupt_mutex_);
        const size_t offset = joint_default_angle_.size() - interrupt_action_.size();
        for (size_t i = 0; i < interrupt_action_.size(); i++) {
            interrupt_action_[i] = static_cast<float>(joint_default_angle_[offset + i]);
        }
    }
    for (PolicyRuntime& policy : policies_) {
        reset_policy_runtime(policy);
    }
    request_depth_history_reset();
}

void InferenceNode::request_depth_history_reset() {
    if (!use_depth_ || !clear_depth_history_client_) {
        return;
    }
    if (!clear_depth_history_client_->service_is_ready()) {
        return;
    }
    auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
    clear_depth_history_client_->async_send_request(request);
}

InferenceNode::PolicyRuntime& InferenceNode::active_policy() {
    return policies_[active_policy_idx_];
}

void InferenceNode::initialize_runtime_state() {
    active_policy_idx_ = 0;

    joint_state_msg_.name.resize(joint_num_);
    joint_state_msg_.position.assign(joint_num_, 0.0f);
    joint_state_msg_.velocity.assign(joint_num_, 0.0f);
    joint_state_msg_.effort.assign(joint_num_, 0.0f);
    action_msg_.name.resize(joint_num_);
    action_msg_.position.assign(joint_num_, 0.0f);
    for (int i = 0; i < joint_num_; i++) {
        joint_state_msg_.name[i] = "joint_" + std::to_string(i + 1);
        action_msg_.name[i] = "action_" + std::to_string(i + 1);
    }

    cmd_vel_.assign(3, 0.0f);
    act_.assign(joint_num_, 0.0f);
    last_act_.assign(joint_num_, 0.0f);
    joint_pos_buffer_.assign(joint_num_, 0.0f);
    joint_vel_buffer_.assign(joint_num_, 0.0f);
    joint_torques_buffer_.assign(joint_num_, 0.0f);
    quat_buffer_.assign(4, 0.0f);
    ang_vel_buffer_.assign(3, 0.0f);
    if (has_obs_source("perception")) {
        perception_obs_buffer_.assign(perception_obs_num_, 0.0f);
    } else {
        perception_obs_buffer_.clear();
    }
    if (has_obs_source("interrupt")) {
        interrupt_action_.assign(10, 0.0f);
    } else {
        interrupt_action_.clear();
    }
}

bool InferenceNode::has_motion_policy() const {
    return !motion_policy_indices_.empty();
}

bool InferenceNode::supports_interrupt() const {
    return !interrupt_action_.empty();
}

void InferenceNode::reset_policy_runtime(PolicyRuntime& policy) {
    std::fill(policy.obs.begin(), policy.obs.end(), 0.0f);
    for (auto& segment : policy.obs_segments) {
        std::fill(segment.begin(), segment.end(), 0.0f);
    }
    if (policy.ctx) {
        std::fill(policy.ctx->input_buffer.begin(), policy.ctx->input_buffer.end(), 0.0f);
        std::fill(policy.ctx->output_buffer.begin(), policy.ctx->output_buffer.end(), 0.0f);
    }
    policy.motion_frame = 0;
    if (policy.latent_loader) {
        policy.latent_loader->reset();
    }
    policy.is_first_frame = true;
}

void InferenceNode::apply_action() {
    std::unique_lock<std::mutex> control_lock(control_mutex_);
    // ⭐ 温度轮询（2026-09-22）：放在所有分支【之前】。
    //   ① 它跟"策略跑不跑"无关 —— 电机只要使能着就值得看温度
    //      （操作员按 B 暂停之后，PD 保底照样在发热）；
    //   ② 过热要做的两件事（切 PD / 失能）必须赶在本周期下发之前生效。
    check_motor_temperature();
    // ⭐ PD 站立保底（2026-09-17）：PD 模式【绕过 is_running_】。
    //
    //   为什么：is_running_ 的语义是"策略要不要跑"（操作员按 B 暂停时会置 false）。
    //   而 PD 站立是【兜底】—— 它不该依赖"策略是否在跑"：
    //     ① 出事降级时，策略可能已被暂停 ⇒ 若也走 is_running_ 检查，则 PD 也下发不出去 ⇒ 仍然瘫软
    //     ② 操作员主动按 B 暂停时，他要的是"松掉" ⇒ 不该被 PD 覆盖
    //   ⇒ 所以把两条路分开：POLICY 模式仍受 is_running_ 管；PD_STAND 模式独立下发。
    //   ⚠️ 副作用（已知且刻意）：进入 PD 后，按 B 暂停不会让电机松掉。要松掉请按 X 失能。
    const bool pd_mode = (act_mode_.load() == ActMode::PD_STAND);
    if (!pd_mode && !is_running_.load()) {
        return;
    }
    bool bad_action = false;
    bool clipped_action = false;
    bool clip_logged_once = false;
    int clipped_joint = -1;
    double clipped_from = 0.0;
    double clipped_to = 0.0;
    {
        std::unique_lock<std::mutex> lock(act_mutex_);
        // ⭐ PD 目标由【本线程】写（2026-09-22）。
        //   原来是推理线程在 PD 模式下写 act_ = joint_default_angle_，
        //   而 control 线程只负责平滑 ⇒ PD 保底依赖"推理线程还活着"。
        //   恰恰在最需要它的时候不成立：
        //     ① 按 B 暂停后跌倒：推理线程过不了 is_running_ 那道门，
        //        永远走不到 PD 分支 ⇒ act_ 停在最后一个策略动作上；
        //     ② 推理 overrun / 卡在 ONNX Run 里：同上。
        //   本线程本来就 250 Hz 在跑，写 act_ 是顺手的事，且不依赖任何人。
        //   ⚠️ 因此推理线程的 PD 分支【不能再写 act_】（两个线程频率不同会互相覆盖）。
        if (pd_mode) {
            for (size_t i = 0; i < act_.size() && i < pd_stand_target_.size(); ++i) {
                act_[i] = static_cast<float>(pd_stand_target_[i]);
            }
        }
        // ⭐ 安全注入（2026-09-22）：放在 PD 目标之后、非有限检查【之前】——
        //   注入的值必须和非策略来源的值走完全一样的后续处理，否则等于绕过了
        //   本来要验证的那段代码。只吃一次：每设一次参数注入一个周期。
        if (inject_pending_.exchange(false)) {
            for (size_t i = 0; i < act_.size() && i < injected_action_.size(); ++i) {
                act_[i] = static_cast<float>(injected_action_[i]);
            }
        }
        // ⭐ 注入 NaN（2026-09-22）：只把指定关节设成 NaN、其它保持原样 ——
        //   这才是"策略某一个输出变 NaN"的真实样子，也正是要验证
        //   下面那段非有限检查能不能接住它。
        const int nan_joint = inject_nan_joint_.exchange(-1);
        if (nan_joint >= 0 && nan_joint < static_cast<int>(act_.size())) {
            act_[nan_joint] = std::numeric_limits<float>::quiet_NaN();
        }
        // ⭐ 非有限目标（2026-09-22）：策略发散时输出会变 NaN，而 NaN 是【粘住】的
        //   —— clip_actions 对 NaN 是空操作，NaN 经观测 last_action 又喂回网络，
        //   之后每一帧都是 NaN。所以不能只记一笔日志，必须当场切 PD。
        //   处理顺序很关键：必须在平滑之前替换，否则 NaN 会先把 last_act_ 污染掉
        //   （last_act_ = α·NaN + (1−α)·last_act_ = NaN），那份状态恢复不回来。
        for (size_t i = 0; i < act_.size(); ++i) {
            if (!std::isfinite(act_[i])) {
                bad_action = true;
                act_[i] = last_act_[i];  // 保持上一帧目标：这一周期不动这个关节
            }
        }
        for (size_t i = 0; i < act_.size(); i++) {
            last_act_[i] = act_alpha_ * act_[i] + (1 - act_alpha_) * last_act_[i];
        }
        // ⭐ 下发目标裁剪（2026-09-22）：见 utils/safety_clip.hpp 的说明。
        //   裁在【关节坐标系】、robot_->apply_action() 之前 —— 踝关节 decouple
        //   在那边内部做，它收到的是关节角。
        //   尺寸不符时【不裁】而不是瞎裁：load_config 已经按 2×joint_num 校验过，
        //   走到这里说明配置为空（= 原本就不裁）。
        if (joint_limits_.size() == 2 * last_act_.size()) {
            for (size_t j = 0; j < last_act_.size(); ++j) {
                const double before = last_act_[j];
                double target = before;
                if (safety::clip_joint_target(target, joint_limits_[2 * j],
                                              joint_limits_[2 * j + 1],
                                              safety::kJointLimitMargin)) {
                    clipped_action = true;
                    if (!clip_logged_once) {  // 记下第一个被裁的关节，供日志用
                        clip_logged_once = true;
                        clipped_joint = static_cast<int>(j);
                        clipped_from = before;
                        clipped_to = target;
                    }
                    last_act_[j] = static_cast<float>(target);
                }
            }
        }
    }
    // 日志都放在锁外（构造字符串不该占着 act_mutex_）
    if (bad_action) {
        // ⚠️ 这里不能只在"从策略切过来"时报（上面几处降级是那个写法）：
        //   非有限值是【数据事件】不是模式迁移 —— PD 模式下同样会发生
        //   （pd_stand_target 配错、或注入进来的值就是 NaN），
        //   而那时 act_mode_ 本来就是 PD，按"模式变了没"判会一个字都不打。
        //   ⇒ 改按时间限速：真的持续坏下去也只会每 2 s 一行。
        const int64_t now_ns = steady_ns();
        if (now_ns - last_bad_action_log_ns_ > 2000000000LL) {
            last_bad_action_log_ns_ = now_ns;
            RCLCPP_ERROR(this->get_logger(),
                         "收到非有限的下发目标（NaN/inf）⇒ 该关节保持上一帧、整体切 PD 站立。"
                         "NaN 会经 last_action 观测自锁，不会自己恢复");
        }
        switch_to_pd_stand("non-finite action");
    }
    if (clipped_action) {
        const int64_t now_ns = steady_ns();
        if (now_ns - last_clip_log_ns_ > 2000000000LL) {  // 每 2 s 最多一行
            last_clip_log_ns_ = now_ns;
            RCLCPP_WARN(this->get_logger(),
                        "下发目标被关节限位裁剪：joint %d  %.3f → %.3f rad"
                        "（限位 [%.3f, %.3f]，内侧留 %.2f）。"
                        "偶发 = 保护生效；常态 = 策略发散或 joint_limits 配错",
                        clipped_joint, clipped_from, clipped_to,
                        joint_limits_[2 * clipped_joint], joint_limits_[2 * clipped_joint + 1],
                        safety::kJointLimitMargin);
        }
    }
    // ⭐ 前馈力矩（2026-09-21）：按 q_des 查表插值，连同位置一起下发。
    //   为什么需要：τ = Kp·(q_des − q) 的关节必须【下垂一点】才有力矩 ⇒ 稳态误差是必然的。
    //   加了前馈，那部分提前给了，Kp 只需修零头。
    //   ⚠️ 符号：这里给【关节坐标】的 τ，motor_sign 由 RobotInterface::motors_mit_cmd
    //      内部乘（和位置同一个规则：τ_m = sign × τ_j）。
    //   ⚠️ enabled=false 或表为空时 ff_tau_ 全 0 ⇒ 与原行为逐位等价。
    //   ⚠️ 表的条件（悬挂/落地）必须和当前运行条件一致 —— 见 robot.yaml 那段注释。
    update_feedforward(last_act_);
    robot_->apply_action(last_act_, {}, {}, {}, ff_tau_);

    // ⭐ 电机离线降级（2026-09-21）
    //   为什么必须切：离线后 motor->get_motor_pos() 返回【上一帧缓存】，
    //   观测会冻住 ⇒ 策略吃不变的关节角 ⇒ 给出无意义的动作。不是因为"电机坏了"才切。
    //
    //   为什么这里就能救回来：RobotInterface::read_joints 现在收 strict=false，
    //   离线【不再抛】⇒ 不会再走 control() 的 rclcpp::shutdown() 分支 ⇒
    //   控制线程活着，活着的电机继续收 PD 帧。
    //
    //   ⚠️ 不刷屏：switch_to_pd_stand 只在 act_mode 真的变化时打日志，
    //      已经在 PD_STAND 时是静默 no-op。字符串也只在真切换那一刻才拼。
    //   ⚠️ 不自动切回：与跌倒检测一致 —— 反复横跳每次都有动作跳变。
    if (robot_->is_init_.load() && robot_->motors_offline()) {
        if (act_mode_.load() == ActMode::POLICY) {
            // ⭐ A2：离线名单进事件（事后看 bag 才知道是哪几台、几次没回）
            const std::string offline = robot_->offline_motor_list();
            switch_to_pd_stand("motor offline", offline);
            RCLCPP_WARN(this->get_logger(), "Offline motors: %s", offline.c_str());
        } else {
            switch_to_pd_stand("motor offline");
        }
    }
}

// ⭐ 温度轮询（2026-09-22）
//   为什么在 control 线程：它是唯一一直在跑的那个（250 Hz），而且过热要做的
//   两件事（切 PD / 失能）都必须赶在下发【之前】生效 —— 放在推理线程会晚一整个周期，
//   且推理线程本身可能就是卡住的那个。
//   开销：10 次 atomic 读，可忽略；但动作仍要连续 kTempConsecutiveCycles 个周期
//   才生效，用来滤掉单帧垃圾（温度 100 ms 内不可能真变）。
// ⭐ 安全注入钩子（2026-09-22）—— A4 用，见头文件里的设计说明。
//
//   ⚠️ 这里是【拒绝点】：总闸关着的时候，设 safety_inject_action / stall 会被
//      参数的 set 回调直接拒掉（ros2 param set 会返回失败原因），而不是静默生效。
//      "关掉的安全功能"和"根本没有的安全功能"要能区分开 —— 这正是 A1① 的教训。
rcl_interfaces::msg::SetParametersResult InferenceNode::on_parameter_change(
    const std::vector<rclcpp::Parameter>& params) {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;

    // ⚠️ 先单独把总闸挑出来算好：同一批里可能既设了总闸又设了注入值，
    //    而"哪个先"不该影响结果（下面那个循环里再读成员就变成顺序相关了）。
    // ⚠️ 而且【必须真的把值写回成员】—— 第一版只打了日志没赋值，
    //    于是运行时开总闸永远开不上（静默不生效）。
    bool enabled = safety_inject_enabled_;
    for (const auto& param : params) {
        if (param.get_name() == "safety_inject_enabled") {
            enabled = param.as_bool();
        }
    }
    if (enabled != safety_inject_enabled_) {
        safety_inject_enabled_ = enabled;
        if (!enabled) {  // 关闸 ⇒ 顺手把还没生效的注入清掉
            inject_stall_ms_.store(0.0f);
            inject_pending_.store(false);
            inject_nan_joint_.store(-1);
        }
        RCLCPP_ERROR(this->get_logger(),
                     "⚠️ safety_inject_enabled -> %s（这之后注入参数才会被接受）",
                     enabled ? "true" : "false");
    }

    for (const auto& param : params) {
        const std::string& name = param.get_name();
        if (name == "safety_inject_enabled") {
            continue;  // 上面处理过了
        }
        if (name == "safety_inject_action") {
            if (!enabled) {
                result.successful = false;
                result.reason = "safety_inject_enabled is false";
                return result;
            }
            const auto values = param.as_double_array();
            if (values.size() != static_cast<std::size_t>(joint_num_)) {
                result.successful = false;
                result.reason = "safety_inject_action must have exactly " +
                                std::to_string(joint_num_) + " values, got " +
                                std::to_string(values.size());
                return result;
            }
            {
                std::unique_lock<std::mutex> lock(act_mutex_);
                injected_action_.assign(values.begin(), values.end());
                inject_pending_.store(true);
            }
            std::string rendered;
            for (const double value : values) {
                rendered += " " + std::to_string(value);
            }
            RCLCPP_ERROR(this->get_logger(),
                         "⚠️ 安全注入：下一周期 act_ 将被替换为 [%s ]（之后照常走平滑→裁剪→下发）",
                         rendered.c_str());
        } else if (name == "safety_inject_nan_joint") {
            if (!enabled) {
                result.successful = false;
                result.reason = "safety_inject_enabled is false";
                return result;
            }
            const int64_t joint = param.as_int();
            if (joint < -1 || joint >= joint_num_) {
                result.successful = false;
                result.reason = "safety_inject_nan_joint must be -1 (off) or a joint index in [0, " +
                                std::to_string(joint_num_ - 1) + "], got " + std::to_string(joint);
                return result;
            }
            inject_nan_joint_.store(static_cast<int>(joint));
            RCLCPP_ERROR(this->get_logger(),
                         "⚠️ 安全注入：下一周期 act_[%ld] 将被设成 NaN", static_cast<long>(joint));
        } else if (name == "safety_inject_stall_ms") {
            if (!enabled) {
                result.successful = false;
                result.reason = "safety_inject_enabled is false";
                return result;
            }
            const double stall_ms = param.as_double();
            inject_stall_ms_.store(static_cast<float>(stall_ms));
            RCLCPP_ERROR(this->get_logger(),
                         "⚠️ 安全注入：推理循环每周期多睡 %.0f ms（造 overrun）", stall_ms);
        } else if (name == "pd_stand_target") {
            // ⭐ A2 期间补（2026-09-22）：这几个参数是【每周期都会读】的成员，
            //   所以运行时改是能生效的 —— 那就必须真的写回成员，否则就是
            //   "ros2 param set 返回成功、实际什么都没变"（本仓库最反对的那种失败）。
            const auto values = param.as_double_array();
            if (values.size() != static_cast<std::size_t>(joint_num_)) {
                result.successful = false;
                result.reason = "pd_stand_target must have exactly " + std::to_string(joint_num_) +
                                " values, got " + std::to_string(values.size());
                return result;
            }
            for (const double value : values) {
                if (!std::isfinite(value)) {
                    result.successful = false;
                    result.reason = "pd_stand_target must be finite";
                    return result;
                }
            }
            {
                std::unique_lock<std::mutex> lock(act_mutex_);
                pd_stand_target_ = values;
            }
            RCLCPP_ERROR(this->get_logger(), "⚠️ pd_stand_target 已在运行时更新（%zu 个值）",
                         values.size());
        } else if (name == "imu_timeout_s") {
            imu_timeout_s_ = static_cast<float>(param.as_double());
            imu_stale_fired_ = false;  // 改了阈值就允许它重新报一次
            RCLCPP_WARN(this->get_logger(), "imu_timeout_s -> %.3f s", imu_timeout_s_);
        } else if (name == "cmd_timeout_s") {
            cmd_timeout_s_ = static_cast<float>(param.as_double());
            cmd_watchdog_fired_.store(false, std::memory_order_relaxed);
            RCLCPP_WARN(this->get_logger(), "cmd_timeout_s -> %.3f s", cmd_timeout_s_);
        } else if (name == "motor_temp_warn_c") {
            temp_warn_c_ = static_cast<float>(param.as_double());
            temp_warn_cycles_ = 0;
            RCLCPP_WARN(this->get_logger(), "motor_temp_warn_c -> %.0f C", temp_warn_c_);
        } else if (name == "motor_temp_cutoff_c") {
            temp_cutoff_c_ = static_cast<float>(param.as_double());
            temp_cutoff_cycles_ = 0;
            RCLCPP_WARN(this->get_logger(), "motor_temp_cutoff_c -> %.0f C", temp_cutoff_c_);
        } else {
            // ⚠️ 其余参数（action_scale / joint_limits / clip_cmd / obs_layouts / model 路径 …）
            //    只在 load_config 里读一次（declare 那一刻），运行时改【不会生效】。
            //    那就【明确拒绝】，别让它返回成功然后什么都没发生 ——
            //    "静默不生效"是本仓库最贵的一类 bug（NaN 守卫、奖励门、gate 都栽在这上面）。
            result.successful = false;
            result.reason = name + " is only read at startup; restart the node to change it";
            return result;
        }
    }
    return result;
}

void InferenceNode::check_motor_temperature() {
    if (!robot_->is_init_.load()) {
        return;
    }
    if (temp_warn_c_ <= 0.0f && temp_cutoff_c_ <= 0.0f) {
        return;
    }
    const float hottest = robot_->max_motor_temperature();
    temp_warn_cycles_ = (temp_warn_c_ > 0.0f && hottest > temp_warn_c_)
                            ? temp_warn_cycles_ + 1 : 0;
    temp_cutoff_cycles_ = (temp_cutoff_c_ > 0.0f && hottest > temp_cutoff_c_)
                              ? temp_cutoff_cycles_ + 1 : 0;

    if (temp_warn_cycles_ == kTempConsecutiveCycles) {
        const bool was_policy = (act_mode_.load() == ActMode::POLICY);
        if (was_policy) {
            RCLCPP_ERROR(this->get_logger(),
                         "电机温度 %.0f ℃ > %.0f ℃ ⇒ 停止策略、切 PD 站立。过热: %s",
                         hottest, temp_warn_c_, robot_->hot_motor_list(temp_warn_c_).c_str());
        }
        switch_to_pd_stand("motor over-temperature", robot_->hot_motor_list(temp_warn_c_));
    }
    if (temp_cutoff_cycles_ >= kTempConsecutiveCycles && !temp_cutoff_fired_) {
        temp_cutoff_fired_ = true;
        RCLCPP_FATAL(this->get_logger(),
                     "电机温度 %.0f ℃ > %.0f ℃ ⇒ 失能（机器人会失去支撑）。过热: %s",
                     hottest, temp_cutoff_c_, robot_->hot_motor_list(temp_cutoff_c_).c_str());
        try {
            robot_->deinit_motors();
        } catch (const std::exception& e) {
            // 并发失能（操作员正好按了 X）会让 deinit 抛"已经失能" —— 那不是故障。
            // ⚠️ 但异常绝不能漏出去：control() 的外层 catch 会 rclcpp::shutdown()，
            //    过热把一个还活着的节点整个关掉，比过热本身更糟。
            RCLCPP_WARN(this->get_logger(), "过热失能时电机已处于失能状态: %s", e.what());
        }
    }
    // 每 10 s 一行最大值 —— 这行是用来【验证读数本身】的：
    // 上板第一眼就该看到像样的 25~45 ℃，而不是等它误触发才发现 byte7 不是温度。
    const int64_t now_ns = steady_ns();
    if (now_ns - last_temp_log_ns_ > 10000000000LL) {
        last_temp_log_ns_ = now_ns;
        RCLCPP_INFO(this->get_logger(), "电机温度最高 %.0f ℃（%s）",
                    hottest, robot_->hot_motor_list(hottest - 0.5f).c_str());
    }
}

void InferenceNode::control() {
    pthread_setname_np(pthread_self(), "control");
    struct sched_param sp{}; sp.sched_priority = 45;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0) {
        RCLCPP_FATAL(this->get_logger(), "Failed to set realtime priority for control thread");
        rclcpp::shutdown();
        return;
    }
    const auto period = std::chrono::microseconds(static_cast<long long>(dt_ * 1000000));
    auto next_release = std::chrono::steady_clock::now();
    while(rclcpp::ok()){
        next_release += period;
        try {
            // ⭐ PD 站立保底（2026-09-17）：在【控制线程】做最快的跌倒检查。
            //    周期由 dt 决定：robots/dm10/configs/default.yaml 是 dt=0.004 ⇒ 250Hz / 4ms。
            //    （原文写"400Hz, 2.5ms"，与配置对不上，2026-09-21 核实后改正。）
            //    为什么不放在 inference 线程（50Hz, 20ms）：跌倒时推理可能已发散，
            //    而且观测计算本身慢一拍。这里直接读 IMU，不经推理。
            if (act_mode_.load() == ActMode::POLICY) {
                try {
                    auto q = robot_->get_quat();
                    Eigen::Quaternionf q_b2w(q[0], q[1], q[2], q[3]);
                    Eigen::Vector3f g_b = q_b2w.inverse() * Eigen::Vector3f(0.0f, 0.0f, -1.0f);
                    // ⚠️ NaN 检查：IMU 掉线时四元数会全 0 → 归一化出 NaN
                    //    ⇒ 必须【主动切 PD】，因为这时策略的 gravity_b / ang_vel
                    //      也全是 NaN，等于瞎了。
                    //
                    // 🔴 2026-09-22 修：原来写的是 `isfinite(x) && x > T` ——
                    //    对 NaN 完全无效（isfinite(NaN)=false，而 NaN>T 本来也是
                    //    false，加不加守卫行为一样）；对 +∞ 更糟（没守卫时会触发，
                    //    加了反而挡住）。2026-09-21 a3e1519 引入，一直是空操作。
                    //    正解是【取反、用 ||】：非有限值 ⇒ 也要切。
                    if (!std::isfinite(g_b.z()) || g_b.z() > gravity_z_upper_) {
                        const bool finite = std::isfinite(g_b.z());
                        switch_to_pd_stand(
                            finite ? "fall detected (gravity_b.z > threshold)"
                                   : "IMU invalid (gravity_b.z not finite)",
                            finite ? "gravity_b.z = " + std::to_string(g_b.z()) +
                                         ", threshold " + std::to_string(gravity_z_upper_)
                                   : std::string("quaternion not usable"));
                    }
                } catch (const std::exception& e) {
                    switch_to_pd_stand("IMU read failed in safety check", e.what());
                }

                // ⭐ IMU 数据陈旧降级（2026-09-22）
                //   上面那个 NULL 检查管的是"四元数解出非有限值"，管不了这一种：
                //   串口断了以后驱动【不再更新缓存】，get_quat() 一直返回最后一帧的值
                //   —— 有穷、不抛异常、看着完全正常。2026-09-22 台架实测：
                //   拔线后四元数 176 秒一字不差，驱动零报错，策略会拿不动的姿态继续走。
                //   ⇒ 只能靠"驱动最后一次收到帧是什么时候"来判，见 imu_driver.hpp。
                //   ⚠️ age < 0 = 一帧都没收到过，那条路归 NaN 守卫管，这里不重复报。
                const float imu_age_s = robot_->imu_data_age_s();
                if (imu_timeout_s_ > 0.0f && imu_age_s > imu_timeout_s_) {
                    if (!imu_stale_fired_) {
                        imu_stale_fired_ = true;
                        RCLCPP_ERROR(this->get_logger(),
                                     "IMU 已 %.0f ms 没有新数据（阈值 %.0f ms）⇒ 切 PD 站立。"
                                     "注意姿态是【冻住】的不是 NaN，上面的 NaN 守卫不会响",
                                     imu_age_s * 1000.0f, imu_timeout_s_ * 1000.0f);
                    }
                    switch_to_pd_stand("IMU data stale",
                                       std::to_string(static_cast<int>(imu_age_s * 1000.0f)) +
                                           " ms without a new frame");
                }

                // ⭐ 指令看门狗（2026-09-22）：手柄 / `/cmd_vel` 断线 ⇒ 切 PD 站立。
                //    原来【完全没有超时】⇒ 手柄蓝牙一断，机器人保持最后一条速度
                //    指令继续走（已知限制 #8）。
                //    ⚠️ 不把速度清零：策略只见过 vx∈[0.3,0.5]，清零是分布外。
                //    切 PD 后 act_mode_ 变了，上面的 if 会把这段整个跳过 ⇒ 只触发一次。
                const int64_t last_cmd = last_cmd_vel_ns_.load(std::memory_order_relaxed);
                if (cmd_timeout_s_ > 0.0f && last_cmd > 0) {
                    const double age_s = static_cast<double>(steady_ns() - last_cmd) * 1e-9;
                    if (age_s > cmd_timeout_s_) {
                        if (!cmd_watchdog_fired_.exchange(true)) {
                            RCLCPP_ERROR(this->get_logger(),
                                         "cmd watchdog: 指令已停 %.2f s (> %.2f s) ⇒ 切 PD 站立",
                                         age_s, cmd_timeout_s_);
                        }
                        switch_to_pd_stand("cmd timeout (joystick / cmd_vel silent)",
                                           std::to_string(age_s) + " s since the last command");
                    } else {
                        cmd_watchdog_fired_.store(false, std::memory_order_relaxed);
                    }
                }
            }
            apply_action();
        } catch (const std::exception& e) {
            RCLCPP_FATAL(this->get_logger(), "Exception in control thread: %s", e.what());
            rclcpp::shutdown();
            return;
        }
        auto loop_end = std::chrono::steady_clock::now();
        if (loop_end > next_release) {
            const auto missed_periods = (loop_end - next_release) / period;
            next_release += period * missed_periods;
            if (next_release < loop_end) {
                next_release += period;
            }
        }
        std::this_thread::sleep_until(next_release);
    }
}

void InferenceNode::inference() {
    pthread_setname_np(pthread_self(), "inference");
    const unsigned int total_cores = std::thread::hardware_concurrency();
    const unsigned int cpu_id = total_cores > 1 ? total_cores / 2 : 0;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
        RCLCPP_FATAL(this->get_logger(), "Failed to bind inference thread to Core %u", cpu_id);
        rclcpp::shutdown();
        return;
    }
    struct sched_param sp{}; sp.sched_priority = 35;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0) {
        RCLCPP_FATAL(this->get_logger(), "Failed to set realtime priority for inference thread");
        rclcpp::shutdown();
        return;
    }
    const auto period = std::chrono::microseconds(static_cast<long long>(dt_ * 1000 * 1000 * decimation_));
    auto next_release = std::chrono::steady_clock::now();
    int overrun_streak = 0;  // ⭐ 连续 overrun 计数（2026-09-22），只在 inference 线程里

    while(rclcpp::ok()){
        next_release += period;
        auto loop_start = std::chrono::steady_clock::now();
        // ⭐ PD 站立保底（2026-09-17）：PD 模式下【不跑网络】。
        //    ⚠️ 2026-09-22 改：act_ 不再由这里写 —— 改由 control 线程在
        //       apply_action() 里写（它 250 Hz 一直在跑，不依赖推理线程活着）。
        //       两处都写会在 pd_stand_target_ ≠ joint_default_angle_ 时互相覆盖：
        //       两个线程频率不同 ⇒ 目标在两个值之间来回跳。
        //       这里只负责把 last_act_ 发到 /action 话题上（观测用）。
        //    ⚠️ 2026-09-22 又改：这段必须放在 is_running_ 那道门【之前】——
        //       规则要和 control 线程里 apply_action() 一致：PD 保底【绕过 is_running_】。
        //       （apply_action 顶部就是这么写的：PD 模式独立下发，不受"策略跑不跑"管。）
        //       原来放在门后，后果是 PD 站立那一段虽然在真发命令，/action 却一条没有
        //       ⇒ bag 里那一段是空的。A2 的验收（/action 与电机实际目标一致）
        //       对这种状态尤其不能漏。
        //       ⚠️ 这个循环里有【两道】is_running_ 检查（门外一道、门内一道，
        //          门内那道是给 mode_mutex_ 竞争用的），别只挪过其中一道。
        if (act_mode_.load() == ActMode::PD_STAND) {
            publish_action();
            const auto now = std::chrono::steady_clock::now();
            if (now > next_release) {
                const auto missed_periods = (now - next_release) / period;
                next_release += period * missed_periods;
                if (next_release < now) {
                    next_release += period;
                }
            }
            std::this_thread::sleep_until(next_release);
            continue;
        }

        if(!is_running_.load()){
            const auto now = std::chrono::steady_clock::now();
            if (now > next_release) {
                const auto missed_periods = (now - next_release) / period;
                next_release += period * missed_periods;
                if (next_release < now) {
                    next_release += period;
                }
            }
            std::this_thread::sleep_until(next_release);
            continue;
        }

        try {
            std::unique_lock<std::mutex> mode_lock(mode_mutex_);
            if (!is_running_.load()) {
                mode_lock.unlock();
                const auto now = std::chrono::steady_clock::now();
                if (now > next_release) {
                    const auto missed_periods = (now - next_release) / period;
                    next_release += period * missed_periods;
                    if (next_release < now) {
                        next_release += period;
                    }
                }
                std::this_thread::sleep_until(next_release);
                continue;
            }

            auto& policy = active_policy();
            // ⭐ 安全注入：造 overrun（2026-09-22）
            //   放在【真正干活的那段】里、PD 分支之后 —— 超时降级只在"策略真的在跑"
            //   时才有意义，暂停时不该触发（也就无从验证它）。见头文件里的钩子说明。
            const float stall_ms = inject_stall_ms_.load();
            if (stall_ms > 0.0f) {
                std::this_thread::sleep_for(
                    std::chrono::microseconds(static_cast<long long>(stall_ms * 1000.0f)));
            }
            robot_->read_imu();
            update_obs_segments(policy.obs_segments, policy.obs_layout);
            publish_imu();
            publish_joint_states();
            flatten_obs_segments(policy.obs_segments, policy.obs.begin());

            std::transform(policy.obs.begin(), policy.obs.end(), policy.obs.begin(), [this](float val) {
                return std::clamp(val, -clip_observations_, clip_observations_);
            });

            if (!policy.history_gather_plan.empty()) {
                if (policy.is_first_frame && policy.latent_loader) {
                    std::fill(policy.obs_history.begin(),
                              policy.obs_history.end() - policy.obs_num, 0.0f);
                    std::copy(policy.obs.begin(), policy.obs.end(),
                              policy.obs_history.end() - policy.obs_num);
                } else {
                    update_obs_history(policy.obs_history, policy.obs, policy.obs_num,
                                       policy.frame_stack, policy.is_first_frame);
                }
                gather_sparse_obs_history(policy.ctx->input_buffer, policy.obs_history,
                                          policy.history_gather_plan);
            } else {
                update_stacked_obs(policy.ctx->input_buffer, policy.obs, policy.obs_num,
                                   policy.frame_stack, policy.stack_order,
                                   policy.obs_layout_sizes, policy.is_first_frame);
            }
            if (policy.motion_loader) {
                step_motion_frame();
            }
            policy.is_first_frame = false;

            policy.ctx->session->Run(Ort::RunOptions{nullptr},
                policy.ctx->input_names_raw.data(), policy.ctx->input_tensor.get(), policy.ctx->num_inputs,
                policy.ctx->output_names_raw.data(), policy.ctx->output_tensor.get(), policy.ctx->num_outputs);

            {
                std::unique_lock<std::mutex> interrupt_lock(interrupt_mutex_, std::defer_lock);
                if (supports_interrupt() && is_interrupt_.load()) {
                    interrupt_lock.lock();
                }
                std::unique_lock<std::mutex> lock(act_mutex_);
                for (int i = 0; i < static_cast<int>(policy.ctx->output_buffer.size()); i++) {
                    policy.ctx->output_buffer[i] = action_rescale_ * std::clamp(
                        policy.ctx->output_buffer[i], -clip_actions_, clip_actions_);
                    const auto joint_idx = usd2urdf_[i];
                    act_[joint_idx] = policy.ctx->output_buffer[i] * action_scale_[joint_idx] +
                                      joint_default_angle_[joint_idx];
                }
                if (interrupt_lock.owns_lock()) {
                    for (size_t i = 0; i < interrupt_action_.size(); i++) {
                        act_[act_.size() - interrupt_action_.size() + i] = interrupt_action_[i];
                    }
                }
            }
            publish_action();
        } catch (const std::exception& e) {
            RCLCPP_FATAL(this->get_logger(), "Exception in inference thread: %s", e.what());
            rclcpp::shutdown();
            return;
        }

        auto loop_end = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(loop_end - loop_start);
        if (loop_end > next_release) {
            // ⭐ 推理超时降级（2026-09-22）：原来只打一行 WARN 就接着跑。
            //   一次超时不必当真（调度抖动），但连续 kOverrunStreakToPd 次
            //   （5 × 20 ms ≈ 100 ms）就说明这台机器已经跟不上控制周期了 ——
            //   此时算出来的动作是"过期"的，继续下发比切 PD 危险。
            //   切过去之后 PD 分支不跑网络，本身也把负载降了。
            //   ⚠️ 判据用 == 不用 >=：只在跨过阈值那一刻报一次，不刷屏。
            if (++overrun_streak == kOverrunStreakToPd) {
                RCLCPP_ERROR(this->get_logger(),
                             "推理连续 %d 次 overrun（本次 %lld us / 周期 %lld us）⇒ 切 PD 站立",
                             overrun_streak, static_cast<long long>(elapsed_time.count()),
                             static_cast<long long>(period.count()));
                switch_to_pd_stand("inference overrun",
                                   "took " + std::to_string(elapsed_time.count()) +
                                       " us, period " + std::to_string(period.count()) + " us");
            }
            RCLCPP_WARN(this->get_logger(), "Inference loop overran! Took %lld us, but period is %lld us.", static_cast<long long>(elapsed_time.count()), static_cast<long long>(period.count()));
            const auto missed_periods = (loop_end - next_release) / period;
            next_release += period * missed_periods;
            if (next_release < loop_end) {
                next_release += period;
            }
        } else {
            overrun_streak = 0;
        }
        std::this_thread::sleep_until(next_release);
    }
}

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    if (mlockall(MCL_CURRENT | MCL_FUTURE) == -1) {
        RCLCPP_WARN(rclcpp::get_logger("main"), "mlockall failed.");
    }
    pthread_setname_np(pthread_self(), "main");
    std::shared_ptr<InferenceNode> node;
    try {
        node = std::make_shared<InferenceNode>();
        rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
        executor.add_node(node);
        RCLCPP_INFO(node->get_logger(), "Press 'X' to initialize/deinitialize motors");
        RCLCPP_INFO(node->get_logger(), "Press 'A' to reset motors");
        RCLCPP_INFO(node->get_logger(), "Press 'B' to start/pause inference");
        RCLCPP_INFO(node->get_logger(), "Press 'Y' to switch between Gamepad Control / cmd_vel Control");
        if (node->supports_interrupt() || node->has_motion_policy()){
            RCLCPP_INFO(node->get_logger(), "Press 'LB' to switch policy mode (available in beyondmimic / interrupt modes)");
        }
        if (node->has_motion_policy()){
            RCLCPP_INFO(node->get_logger(), "Press 'RB' to switch motion sequence (available in beyondmimic mode)");
        }
        RCLCPP_INFO(node->get_logger(), "Right Stick: Control forward, backward, left and right movement");
        RCLCPP_INFO(node->get_logger(), "LT/RT: Control turning (left / right rotation)");
        executor.spin();
    } catch (const std::exception &e) {
        RCLCPP_FATAL(rclcpp::get_logger("main"), "Exception caught: %s", e.what());
    }
    rclcpp::shutdown();
    node.reset();
    return 0;
}
