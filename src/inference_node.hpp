// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#pragma once

#include <sys/mman.h>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <Eigen/Geometry>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <queue>
#include <sstream>
#include <thread>
#include <utility>          // std::pair —— 前馈表用
#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/string.hpp>          // ⭐ A2 事件话题（/act_mode）
#include "utils/motion_loader.hpp"
#include "utils/latent_loader.hpp"
#include "utils/safety_clip.hpp"
#include <std_srvs/srv/trigger.hpp>
#include "robot_interface.hpp"

enum class ObsStackOrder {
    FrameMajor,
    ObsMajor,
};

class InferenceNode;

struct ObsSourceDefinition {
    const char* name;
    void (InferenceNode::*get)(std::vector<float>& segment);
};

struct ObsSourceSpec {
    std::string name;
    const ObsSourceDefinition* source;
    int size;
    // Empty keeps legacy contiguous stacking; otherwise values are explicit inference-tick lags.
    std::vector<int> history_taps;
};

struct ObsHistorySlice {
    size_t history_offset;
    size_t size;
};

class InferenceNode : public rclcpp::Node {
   public:
    struct ModelContext {
        std::unique_ptr<Ort::Session> session;
        std::unique_ptr<Ort::MemoryInfo> memory_info;
        std::unique_ptr<Ort::Value> input_tensor;
        std::unique_ptr<Ort::Value> output_tensor;
        std::vector<std::string> input_names;
        std::vector<std::string> output_names;
        std::vector<const char *> input_names_raw;
        std::vector<const char *> output_names_raw;
        std::vector<int64_t> input_shape;
        std::vector<int64_t> output_shape;
        std::vector<float> input_buffer;
        std::vector<float> output_buffer;
        size_t num_inputs;
        size_t num_outputs;
    };

    struct PolicyRuntime {
        std::string name;
        std::string model_path;
        std::string motion_path;
        std::vector<ObsSourceSpec> obs_layout;
        std::vector<int> obs_layout_sizes;
        std::vector<std::vector<float>> obs_segments;
        std::vector<float> obs;
        // Sparse mode only: complete frames ordered from oldest to newest.
        std::vector<float> obs_history;
        // Non-empty selects sparse mode and describes sequential copies into the model input.
        std::vector<ObsHistorySlice> history_gather_plan;
        int obs_num = 0;
        int obs_input_num = 0;
        int frame_stack = 1;
        ObsStackOrder stack_order = ObsStackOrder::FrameMajor;
        std::unique_ptr<ModelContext> ctx;
        std::shared_ptr<MotionLoader> motion_loader;
        std::unique_ptr<LatentLoader> latent_loader;
        size_t motion_frame = 0;
        bool is_first_frame = true;
    };

    InferenceNode() : Node("inference_node") {
        load_config();

        robot_ = std::make_shared<RobotInterface>(robot_config_path_);

        Ort::ThreadingOptions thread_opts;
        if (intra_threads_ > 0) {
            thread_opts.SetGlobalIntraOpNumThreads(intra_threads_);
        }
        env_ = std::make_unique<Ort::Env>(thread_opts, ORT_LOGGING_LEVEL_WARNING, "ONNXRuntimeInference");
        if (policies_.empty()) {
            throw std::runtime_error("At least one policy must be configured");
        }
        for (size_t i = 0; i < policies_.size(); i++) {
            PolicyRuntime& policy = policies_[i];
            policy.obs.resize(policy.obs_num, 0.0f);
            policy.obs_segments.resize(policy.obs_layout.size());
            for (size_t j = 0; j < policy.obs_layout.size(); j++) {
                policy.obs_segments[j].resize(policy.obs_layout[j].size, 0.0f);
            }
            if (!policy.history_gather_plan.empty()) {
                policy.obs_history.resize(
                    static_cast<size_t>(policy.obs_num) * static_cast<size_t>(policy.frame_stack),
                    0.0f);
            }
            if (!policy.motion_path.empty()) {
                policy.motion_loader = std::make_shared<MotionLoader>(policy.motion_path);
                if (policy.motion_loader->get_num_frames() == 0) {
                    throw std::runtime_error("Motion file has no frames: " + policy.motion_path);
                }
                if (policy.motion_loader->get_num_joints() != static_cast<size_t>(joint_num_)) {
                    throw std::runtime_error("Motion joint count mismatch: " + policy.motion_path);
                }
            }
            setup_model(policy.ctx, policy.model_path, policy.obs_input_num);
        }
        initialize_runtime_state();
        reset_runtime_state();

        auto data_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
        joy_subscription_ = this->create_subscription<sensor_msgs::msg::Joy>(
            "/joy", data_qos, std::bind(&InferenceNode::subs_joy_callback, this, std::placeholders::_1));
        cmd_subscription_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", data_qos, std::bind(&InferenceNode::subs_cmd_callback,this, std::placeholders::_1
        ));
        if (has_obs_source("perception")) {
            perception_subscription_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
                perception_obs_topic_, data_qos,
                std::bind(&InferenceNode::subs_perception_callback, this, std::placeholders::_1));
        }
        if (use_depth_) {
            clear_depth_history_client_ =
                this->create_client<std_srvs::srv::Trigger>("clear_depth_history");
        }
        joint_state_subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_ref_states", data_qos,
            std::bind(&InferenceNode::subs_joint_state_callback, this, std::placeholders::_1));
        action_publisher_ =
            this->create_publisher<sensor_msgs::msg::JointState>("/action", data_qos);
        // ⭐ A2（2026-09-22）：模式事件。transient_local ⇒ 录包的人中途接上也能
        //   立刻拿到"现在是什么模式"，不用从零开始等下一次切换。
        auto event_qos = rclcpp::QoS(rclcpp::KeepLast(10)).transient_local().reliable();
        act_mode_publisher_ =
            this->create_publisher<std_msgs::msg::String>("/act_mode", event_qos);
        publish_act_mode("", start_mode_policy_ ? "POLICY" : "PD_STAND", "startup", "");
        // ⭐ A3（2026-09-22）：运行时身份。和 /act_mode 一样 transient_local ——
        //   录包的人中途接上也能立刻知道"这段录像是拿什么跑出来的"。
        node_metadata_publisher_ =
            this->create_publisher<std_msgs::msg::String>("/node_metadata", event_qos);
        publish_node_metadata();
        imu_publisher_ =
            this->create_publisher<sensor_msgs::msg::Imu>("/imu", data_qos);
        // ⭐ A3：模型输入（39 维 @50Hz ≈ 8 kB/s；没人订阅时 best_effort 直接丢，不占带宽）
        obs_publisher_ =
            this->create_publisher<std_msgs::msg::Float32MultiArray>("/policy_obs", data_qos);
        joint_state_publisher_ =
            this->create_publisher<sensor_msgs::msg::JointState>("/joint_states", data_qos);
        inference_thread_ = std::thread(&InferenceNode::inference, this);
        control_thread_ = std::thread(&InferenceNode::control, this);

        reset_joints_service_ = this->create_service<std_srvs::srv::Trigger>(
            "reset_joints", std::bind(&InferenceNode::reset_joints_srv, this, std::placeholders::_1, std::placeholders::_2));
        set_zeros_service_ = this->create_service<std_srvs::srv::Trigger>(
            "set_zeros", std::bind(&InferenceNode::set_zeros_srv, this, std::placeholders::_1, std::placeholders::_2));
        clear_errors_service_ = this->create_service<std_srvs::srv::Trigger>(
            "clear_errors", std::bind(&InferenceNode::clear_errors_srv, this, std::placeholders::_1, std::placeholders::_2));
        refresh_joints_service_ = this->create_service<std_srvs::srv::Trigger>(
            "refresh_joints", std::bind(&InferenceNode::refresh_joints_srv, this, std::placeholders::_1, std::placeholders::_2));
        read_joints_service_ = this->create_service<std_srvs::srv::Trigger>(
            "read_joints", std::bind(&InferenceNode::read_joints_srv, this, std::placeholders::_1, std::placeholders::_2));
        read_imu_service_ = this->create_service<std_srvs::srv::Trigger>(
            "read_imu", std::bind(&InferenceNode::read_imu_srv, this, std::placeholders::_1, std::placeholders::_2));
        init_motors_service_ = this->create_service<std_srvs::srv::Trigger>(
            "init_motors", std::bind(&InferenceNode::init_motors_srv, this, std::placeholders::_1, std::placeholders::_2));
        deinit_motors_service_ = this->create_service<std_srvs::srv::Trigger>(
            "deinit_motors", std::bind(&InferenceNode::deinit_motors_srv, this, std::placeholders::_1, std::placeholders::_2));
        start_inference_service_ = this->create_service<std_srvs::srv::Trigger>(
            "start_inference", std::bind(&InferenceNode::start_inference_srv, this, std::placeholders::_1, std::placeholders::_2));
        stop_inference_service_ = this->create_service<std_srvs::srv::Trigger>(
            "stop_inference", std::bind(&InferenceNode::stop_inference_srv, this, std::placeholders::_1, std::placeholders::_2));
        // ⭐ PD 站立保底（2026-09-17）
        switch_to_pd_service_ = this->create_service<std_srvs::srv::Trigger>(
            "switch_to_pd_stand", std::bind(&InferenceNode::switch_to_pd_srv, this, std::placeholders::_1, std::placeholders::_2));
        switch_to_policy_service_ = this->create_service<std_srvs::srv::Trigger>(
            "switch_to_policy", std::bind(&InferenceNode::switch_to_policy_srv, this, std::placeholders::_1, std::placeholders::_2));
    }
    ~InferenceNode() {
        is_running_.store(false);
        if (reset_thread_.joinable()) {
            reset_thread_.join();
        }
        if (inference_thread_.joinable()) {
            inference_thread_.join();
        }
        if (control_thread_.joinable()) {
            control_thread_.join();
        }
        reset_runtime_state();
        if (robot_) {
            robot_.reset();
        }
    }
    bool supports_interrupt() const;
    bool has_motion_policy() const;

    // ⭐ PD 站立保底 —— 模式枚举（2026-09-17 新增）
    //   PD_STAND：不跑网络，act_ 恒 = joint_default_angle_（僵直站立，可手扶）
    //   POLICY  ：正常跑策略
    //   ⚠️ 默认 PD_STAND（安全优先）。要跑策略必须显式切（service 或手柄）。
    //   ⚠️ 跌倒/关节超限【自动切到 PD_STAND，且不自动切回】——
    //      自动切回会在阈值附近反复横跳，而每次切换都有动作跳变。
    enum class ActMode { PD_STAND, POLICY };
    // ⭐ 事件话题（2026-09-22，A2）：模式切换要能被录下来 —— A3 的 bag 里如果没有
    //   这一串，事后分不清"这段动作是策略跑的"还是"PD 兜底在撑"。
    //   用 std_msgs/String 装一行 JSON（不加自定义 msg：省掉 msg/ + CMake 改动，
    //   而消费方是 Python 工具链，json.loads 最省事）。
    //   QoS 用 transient_local ⇒ 录包的人中途接上也能立刻拿到当前模式。
    void publish_act_mode(const char* prev_mode, const char* new_mode,
                          const std::string& reason, const std::string& detail);
    void switch_to_pd_stand(const char* reason, const std::string& detail = "");
    void switch_to_policy();
    // ⭐ A3（2026-09-22）：把"这段录像是拿什么跑出来的"随 bag 一起留下来。
    //   为什么要有：事后拿到一段录像，第一个问题永远是"这是哪个配置/哪个 onnx/哪版代码"。
    //   ⚠️ 报的是【生效值】（override 之后），不是文件里的值 —— 两者在板上常常不同。
    void publish_node_metadata();
    ActMode act_mode() const { return act_mode_.load(); }
   private:
    std::shared_ptr<RobotInterface> robot_;
    std::atomic<bool> is_running_{false}, is_joy_control_{true}, is_interrupt_{false}, is_motion_policy_{false};
    std::atomic<ActMode> act_mode_{ActMode::PD_STAND};   // ⭐ 默认安全
    bool start_mode_policy_ = false;                     // 启动参数（yaml start_mode: policy 时为 true）
    std::string robot_config_path_;
    std::string robot_name_;          // ⭐ A3：元数据里要报（从参数读，不在局部变量里）
    std::string policy_name_;         // ⭐ A3
    std::string perception_obs_topic_;
    size_t current_motion_policy_idx_ = 0;
    int active_policy_idx_ = 0;
    int perception_obs_num_, joint_num_;
    bool use_depth_ = false;
    int decimation_;
    std::unique_ptr<Ort::Env> env_;
    int intra_threads_;
    Ort::AllocatorWithDefaultOptions allocator_;
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_subscription_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_subscription_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr perception_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_subscription_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr action_publisher_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr act_mode_publisher_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr node_metadata_publisher_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr obs_publisher_;
    std::string policy_config_path_;   // ⭐ A3：policy yaml 的路径（launch 传进来，供哈希用）
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_publisher_;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr clear_depth_history_client_;
    std::thread inference_thread_;
    std::thread control_thread_;
    std::thread reset_thread_;
    std::mutex reset_thread_mutex_;
    bool reset_thread_running_ = false;
    float act_alpha_;
    float dt_;
    float obs_scales_lin_vel_, obs_scales_ang_vel_, obs_scales_dof_pos_, obs_scales_dof_vel_,
        obs_scales_gravity_b_, clip_observations_;
    float clip_actions_;
    float action_rescale_ = 1.0f;
    std::vector<double> action_scale_, clip_cmd_, joint_default_angle_, joint_limits_;

    // ⭐ PD 站立目标（2026-09-22）：和 joint_default_angle_ 分开，因为这是两件事。
    //   joint_default_angle_ = 【策略的参考系】：必须与 sim 的 home keyframe 逐位相等
    //     （obs 零点 / 动作偏置 / 上电软启动都读它），单边改会让策略整体平移。
    //   pd_stand_target_     = 【保底要摆成的姿态】：只求落地能站住，
    //     与策略是屈膝还是直腿无关 —— 这两件事迟早会分叉，绑在一起就改不动。
    //   ⚠️ 留空 = 沿用 joint_default_angle_ ⇒ 与原行为逐位一致。
    std::vector<double> pd_stand_target_;

    // ⭐ 温度轮询（2026-09-22）：线圈温度，取的是 DM 反馈帧 byte7
    //   （dm_motor_driver.cpp:251）。80 ℃ 停策略切 PD、90 ℃ 失能，0 = 关闭该项。
    //   ⚠️ 阈值要连续 kTempConsecutiveCycles 个周期（100 ms）都超才动作 ——
    //      温度不可能在 100 ms 里真变，连续计数滤掉的是单帧垃圾。
    float temp_warn_c_ = 80.0f;
    float temp_cutoff_c_ = 90.0f;
    static constexpr int kTempConsecutiveCycles = 25;   // 25 × dt(4ms) = 100 ms
    // ⭐ 推理超时降级（2026-09-22）：连续这么多次 overrun 就切 PD。
    //   5 × 20 ms（dt × decimation）= 100 ms —— 一次卡顿不切，持续卡就切。
    static constexpr int kOverrunStreakToPd = 5;

    // ↓ 以下计数器【只由 control 线程读写】（overrun 那个在 inference 线程的局部变量里）
    int temp_warn_cycles_ = 0;
    int temp_cutoff_cycles_ = 0;
    bool temp_cutoff_fired_ = false;   // 失能只做一次
    int64_t last_temp_log_ns_ = 0;     // 每 10 s 报一行最热电机（顺便验证读数是否可信）
    int64_t last_clip_log_ns_ = 0;     // 裁剪日志限速
    int64_t last_bad_action_log_ns_ = 0;  // 非有限目标日志限速
    // ⭐ 前馈力矩表（2026-09-21）：τ_ff(q) 查表插值。
    //   来源 robot.yaml 的 gravity_feedforward:（那里的注释写了条件与测法）。
    //   ⚠️ 表的条件（悬挂/落地）必须和当前运行条件一致，否则会【系统性地】喂错力矩。
    bool ff_enabled_ = false;
    std::vector<std::vector<std::pair<double, double>>> ff_table_;  // 每关节 [(q, tau)]，按 q 升序
    std::vector<float> ff_tau_;                                     // 本周期算出的 τ_ff（关节坐标）
    std::vector<long int> usd2urdf_;
    float gravity_z_upper_;
    // ⭐ 指令看门狗（2026-09-22）：手柄 / `/cmd_vel` 断线时，机器人会【保持最后一条
    //   速度指令继续走】—— 全仓原本没有任何超时（已知限制 #8）。
    //   这里记录最后一次收到指令的时刻，超过 cmd_timeout_s_ 就切 PD 站立。
    //
    //   ⚠️ 为什么是【切 PD】而不是【把速度清零】：
    //      训练只喂过 vx ∈ [0.3, 0.5]、`rel_standing_envs: 0.0`
    //      ⇒ 策略【从没见过零命令】，清零反而是分布外，行为不可预测。
    //      切 PD 是绕开策略、走已知可用的那条兜底。
    //   ⚠️ 默认取 1.0 而不是 0.5：误触发的代价是【行走中途突然切 PD】，
    //      而漏触发的代价只是"晚半秒刹车"。且工具箱里有以低频发 `/cmd_vel`
    //      的脚本（如 sweep_cmd_vx.py），0.5 s 会把正常发帧当成断线。
    //      跑通一次、确认不误触发之后再收到 0.5；用脚本低频驱动时设 0。
    float cmd_timeout_s_ = 1.0f;
    std::atomic<int64_t> last_cmd_vel_ns_{0};       // steady_clock 纳秒
    std::atomic<bool> cmd_watchdog_fired_{false};   // 只在状态翻转时打日志，不刷屏

    // ⭐ IMU 数据陈旧降级（2026-09-22）
    //   串口断掉之后 get_quat() 会一直返回【最后一帧的缓存值】：不抛异常、不是 NaN，
    //   所以 A1① 那个 NaN 守卫根本不触发，而策略拿的是不动的姿态继续走。
    //   2026-09-22 台架实测：拔线后四元数 176 秒一字不差、驱动零报错、RX 线程还吃满一个核。
    //   ⚠️ 判"驱动还有没有在收帧"（时间戳由驱动盖），不是判"数值有没有变"
    //      —— 静止时数值本来就可能连续几帧完全相同，那样会误触发。
    //   ⚠️ 默认 0.05 s：实测 IMU 是 ~1000 Hz（1 ms 一帧），50 ms = 漏 50 帧，
    //      不存在误触发；再紧也能调（0 = 关闭）。
    float imu_timeout_s_ = 0.05f;
    bool imu_stale_fired_ = false;                  // 只由 control 线程读写

    // ⭐ 安全注入钩子（2026-09-22）：给 A4 注入测试用，默认全关。
    //
    //   为什么需要：有些故障【只能从程序内部发生】——
    //     · `act_` 是策略在节点内部算出来的，外面没有任何话题能往里塞一个 NaN 或超限值；
    //     · overrun 是"这一轮没算完"，推理线程是 SCHED_FIFO 35，外面压不垮它。
    //   没有钩子，这两条安全代码就只能靠读代码验证 —— 而"写了但从不触发"正是
    //   A1① 那个 NaN 守卫死掉的方式（对 NaN 是空操作，在仓库里躺了四天）。
    //
    //   为什么用【参数】而不是话题：`ros2 param set` 要一个字一个字敲，误触发概率最低；
    //   也不用加 .srv 定义和编译改动。连续值（动作向量）正好能用 double 数组参数表达。
    //
    //   ⚠️ 注入值是【替换 act_】，之后和策略动作走完全相同的路（平滑 → 裁剪 → 下发）
    //      ⇒ 验的是真代码路径，不是旁路。
    //   ⚠️ 总闸 safety_inject_enabled 默认 false：部署时的启动参数一个字都不用改，
    //      关着的时候下面两个参数设了也会被参数的 set 回调【拒绝】。
    bool safety_inject_enabled_ = false;
    std::vector<double> injected_action_;           // 受 act_mutex_ 保护
    std::atomic<bool> inject_pending_{false};
    std::atomic<float> inject_stall_ms_{0.0f};
    // NaN 单独一个参数，不走"数组里放 nan"：ROS 的 C 版 YAML 解析器吃不下
    // 数组里的 `.nan`（整条会被当成字符串，param load 报类型错）。
    // 语义：>=0 ⇒ 只把该关节的 act_ 设成 NaN，其它关节保持原样 ——
    //       这才是"策略某一个输出变 NaN"的真实样子。
    std::atomic<int> inject_nan_joint_{-1};
    OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;
    rcl_interfaces::msg::SetParametersResult on_parameter_change(
        const std::vector<rclcpp::Parameter>& params);

    // 单调时钟纳秒（看门狗不能用 ROS time —— 那是可跳的仿真时间）
    static int64_t steady_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }
    int last_button0_ = 0, last_button1_ = 0, last_button2_ = 0, last_button3_ = 0, last_button4_ = 0, last_button5_ = 0;
    std::vector<PolicyRuntime> policies_;
    std::vector<int> motion_policy_indices_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_joints_service_, set_zeros_service_, clear_errors_service_, refresh_joints_service_, read_joints_service_, read_imu_service_, init_motors_service_, deinit_motors_service_, start_inference_service_, stop_inference_service_,\
        switch_to_pd_service_, switch_to_policy_service_;

    std::mutex act_mutex_, perception_mutex_, interrupt_mutex_, cmd_mutex_, mode_mutex_, control_mutex_, lb_switch_mutex_;
    std::vector<float> act_, last_act_, cmd_vel_, interrupt_action_, perception_obs_buffer_;
    std::vector<float> joint_pos_buffer_, joint_vel_buffer_, joint_torques_buffer_, quat_buffer_, ang_vel_buffer_;
    sensor_msgs::msg::JointState joint_state_msg_, action_msg_;
    std_msgs::msg::Float32MultiArray obs_msg_;   // ⭐ A3：给 onnx 的那份输入

    void subs_joy_callback(const std::shared_ptr<sensor_msgs::msg::Joy> msg);
    void subs_cmd_callback(const std::shared_ptr<geometry_msgs::msg::Twist> msg);
    void subs_perception_callback(const std::shared_ptr<std_msgs::msg::Float32MultiArray> msg);
    void subs_joint_state_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg);
    void inference();
    void control();
    void apply_action();
    bool start_joint_reset();
    PolicyRuntime& active_policy();

    void load_config();
    void load_feedforward_table();
    void update_feedforward(const std::vector<float>& q_des);
    void check_motor_temperature();
    void setup_model(std::unique_ptr<ModelContext>& ctx, std::string model_path, int input_size);

    // Policy/model runtime helpers.
    void initialize_runtime_state();
    void reset_runtime_state();
    void request_depth_history_reset();
    void reset_policy_runtime(PolicyRuntime& policy);
    void step_motion_frame();

    // Observation registry and layout helpers.
    static const std::vector<ObsSourceDefinition>& obs_source_definitions();
    std::vector<ObsSourceSpec> parse_obs_layout(const std::string& layout_spec,
                                                const std::string& layout_name);
    bool has_obs_source(const std::string& source_name) const;
    ObsStackOrder parse_obs_stack_order(const std::string& stack_order_name);

    // Observation runtime helpers.
    void update_obs_segments(std::vector<std::vector<float>>& segments,
                             const std::vector<ObsSourceSpec>& layout);
    void flatten_obs_segments(const std::vector<std::vector<float>>& segments,
                              std::vector<float>::iterator output_begin);
    void update_obs_history(std::vector<float>& history, const std::vector<float>& obs,
                            int obs_num, int frame_stack, bool is_first_frame);
    void update_stacked_obs(std::vector<float>& input_buffer, const std::vector<float>& obs,
                            int obs_num, int frame_stack, ObsStackOrder stack_order,
                            const std::vector<int>& field_sizes, bool is_first_frame);
    void gather_sparse_obs_history(std::vector<float>& input_buffer,
                                   const std::vector<float>& obs_history,
                                   const std::vector<ObsHistorySlice>& gather_plan);

    // Observation getters.
    void get_cmd_vel_obs(std::vector<float>& segment);
    void get_ang_vel_obs(std::vector<float>& segment);
    void get_gravity_b_obs(std::vector<float>& segment);
    void get_dof_pos_obs(std::vector<float>& segment);
    void get_dof_vel_obs(std::vector<float>& segment);
    void get_last_action_obs(std::vector<float>& segment);
    void get_interrupt_obs(std::vector<float>& segment);
    void get_perception_obs(std::vector<float>& segment);
    void get_motion_command_obs(std::vector<float>& segment);
    void get_latent_obs(std::vector<float>& segment);

    void init_motors_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                         std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void deinit_motors_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                           std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void reset_joints_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                          std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void set_zeros_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                       std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void clear_errors_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                          std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void refresh_joints_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                            std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void read_joints_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                         std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void read_imu_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                      std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void start_inference_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                             std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void switch_to_pd_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                          std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void switch_to_policy_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                              std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void stop_inference_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                            std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    // ⭐ A3（2026-09-23）：一拍只读一次关节状态。
    //   原来 obs 算一次（getter 里读）、发 /joint_states 又读一次，
    //   中间隔着一次 DDS 发布；而 control 线程每 4 ms 刷新缓存
    //   ⇒ 约 28% 的拍，bag 里的 /joint_states 是【下一拍】的快照。
    //   验收脚本就是这么抓出来的（dof_pos/dof_vel 28% 帧差一个控制周期）。
    //   现在：每拍开头读一次进缓冲，obs 与 /joint_states 都用这一份。
    void snapshot_joint_state();
    void publish_joint_states();
    void publish_action();
    void publish_imu();
    // ⭐ A3（2026-09-23）：把【喂给 onnx 的那份输入】发出来。
    //   为什么必须发：A3 的验收是"离线重建的 obs 与真机逐帧一致（<1e-6）"，
    //   而 obs 原本哪儿都没落盘 ⇒ 只能隔着 /action 比，而 /action 又隔着
    //   act_alpha 平滑和 250/50 Hz 采样差，永远比不到 1e-6。
    //   发 input_buffer 而不是 policy.obs：前者是过完 obs 裁剪、过完 frame_stack
    //   堆叠之后真正送进模型的那份，可以逐位比。
    void publish_obs();
    
    template <typename T>
    void print_vector(const std::string& name, const std::vector<T>& vec) {
        std::stringstream ss;
        ss << name << ": [";
        for (size_t i = 0; i < vec.size(); ++i) {
            ss << vec[i] << (i == vec.size() - 1 ? "" : ", ");
        }
        ss << "]";
        RCLCPP_INFO(this->get_logger(), "%s", ss.str().c_str());
    }
};
