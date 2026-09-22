#pragma once

#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <algorithm>
#include <memory>
#include <Eigen/Geometry>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <queue>
#include <sstream>
#include <yaml-cpp/yaml.h>
#include "utils/close_chain_mapping.hpp"
#include "utils/thread_pool.hpp"
#include "motor_driver.hpp"
#include "imu_driver.hpp"

class RobotInterface {
   public:
    RobotInterface(const std::string& config_file);
    ~RobotInterface() {
        if (is_init_.load()) {
            deinit_motors();
        }
        motors_.clear();
        imu_.reset();
    }
    struct IMUCfg{
        int imu_id_, baudrate_;
        std::string imu_type_, imu_interface_type_, imu_interface_;
    };
    struct MotorsCfg{
        int master_id_offset_;
        std::vector<std::string> motor_type_;
        std::vector<std::string> motor_interface_type_;
        std::vector<std::string> motor_interface_;
        std::vector<long int> motor_id_, motor_model_, motor_num_;
        std::vector<double> motor_zero_offset_;
    };
    struct RobotCfg{
        std::vector<long int> close_chain_motor_idx_, motor_sign_, urdf2motor_;
        std::vector<double> kp_, kd_, extrinsic_R_;
    };

    void apply_action(const std::vector<float>& p,
                      const std::vector<float>& v = {},
                      const std::vector<float>& kp = {},
                      const std::vector<float>& kd = {},
                      const std::vector<float>& tau = {});
    void init_motors();
    void deinit_motors();
    void reset_joints(std::vector<double> joint_default_angle);
    void set_zeros();
    void clear_errors();
    // strict=false：电机未使能时【静默返回】、离线时【不抛】—— 给 control 线程用。
    //   ⚠️ 为什么需要：read_joints() 在 motors_mit_cmd() 【之前】，一抛就本周期一帧都发不出去；
    //      而 control() 的外层 catch 会 rclcpp::shutdown() ⇒ 之后永远不再发帧 ⇒ 机器人瘫倒。
    //      true（默认）保持原语义，给 reset_joints / refresh_joints / read_joints 服务用。
    void read_joints(bool strict = true);
    void read_imu();
    void refresh_joints();
    std::vector<float> get_joint_q() {
        if (!is_init_.load()) {
            throw std::runtime_error("Motors are not initialized");
        }
        std::unique_lock<std::mutex> lock(joint_mutex_);
        return joint_q_;
    }
    std::vector<float> get_joint_vel() {
        if (!is_init_.load()) {
            throw std::runtime_error("Motors are not initialized");
        }
        std::unique_lock<std::mutex> lock(joint_mutex_);
        return joint_vel_;
    }
    std::vector<float> get_joint_tau() {
        if (!is_init_.load()) {
            throw std::runtime_error("Motors are not initialized");
        }
        std::unique_lock<std::mutex> lock(joint_mutex_);
        return joint_tau_;
    }
    std::vector<float> get_quat() {
        if (!imu_) {
            throw std::runtime_error("IMU is not initialized");
        }
        std::unique_lock<std::mutex> lock(imu_mutex_);
        return quat_buf_;
    }
    std::vector<float> get_ang_vel() {
        if (!imu_) {
            throw std::runtime_error("IMU is not initialized");
        }
        std::unique_lock<std::mutex> lock(imu_mutex_);
        return ang_vel_buf_;
    }
    /// Linear acceleration in body frame (m/s^2), gravity included.
    /// The DM-IMU-L1 reports ~+9.8 on z when at rest, matching the ROS
    /// sensor_msgs/Imu convention, so no gravity subtraction is applied here.
    std::vector<float> get_lin_acc() {
        if (!imu_) {
            throw std::runtime_error("IMU is not initialized");
        }
        std::unique_lock<std::mutex> lock(imu_mutex_);
        return lin_acc_buf_;
    }

    // ⭐ 电机离线降级（2026-09-21）
    //   ⚠️ 这两个【不抛】—— 它们要在 control 线程（250Hz）的热路径上每周期调用。
    //      抛异常的那条路（throw_if_motors_offline）保留给服务/标定路径。
    //   ⚠️ 只读 atomic，不拿 joint_mutex_ —— 别在这里加锁。
    //   语义：某电机【连续】发送未收到回复的次数 > offline_threshold_ ⇒ 判离线
    //        （response_count_ 每收到一帧就清零，见 dm_motor_driver.cpp:228）
    bool motors_offline() const;
    std::string offline_motor_list() const;

    // ⭐ 温度监控（2026-09-22）：和 motors_offline() 同一类东西 ——
    //   热路径（control 线程 250Hz）用的只读查询，不抛、不加锁。
    //   ⚠️ 单位就是电机反馈里那个数：DM 协议反馈帧 byte7 = 线圈温度（℃）。
    //      没收到过帧的电机读到 0 ⇒ 表现只会是"最大值偏小"，不会误报过热。
    float max_motor_temperature() const;
    std::string hot_motor_list(float limit) const;

    // ⭐ IMU 数据新鲜度（2026-09-22）：见 src/imu/include/imu_driver.hpp 的说明。
    //   返回【秒】；< 0 表示一帧都还没收到过 —— 那不是"陈旧"，由 NaN 守卫负责，
    //   调用方要区分这两种情况。
    //   同样是只读 atomic、不加锁，热路径可每周期调用。
    float imu_data_age_s() const;

    std::atomic<bool> is_init_{false};

   private:
    std::shared_ptr<IMUCfg> imu_cfg_;
    std::shared_ptr<MotorsCfg> motors_cfg_;
    std::shared_ptr<RobotCfg> robot_cfg_;
    int offline_threshold_ = 25;
    std::shared_ptr<IMUDriver> imu_;
    std::shared_ptr<Decouple> ankle_decouple_;
    Eigen::Matrix3f extrinsic_R_mat_ = Eigen::Matrix3f::Identity();
    Eigen::Quaternionf extrinsic_q_inv_ = Eigen::Quaternionf::Identity();
    std::vector<float> quat_buf_{0.f, 0.f, 0.f, 0.f};
    std::vector<float> ang_vel_buf_{0.f, 0.f, 0.f};
    std::vector<float> lin_acc_buf_{0.f, 0.f, 0.f};
    std::vector<std::shared_ptr<MotorDriver>> motors_;
    std::unique_ptr<ThreadPool> thread_pool_;
    std::vector<size_t> motor_bus_offsets_;

    std::mutex reset_mutex_, command_mutex_, motors_mutex_, joint_mutex_, imu_mutex_;
    std::vector<float> joint_q_, joint_vel_, joint_tau_;
    std::vector<float> motor_pos_target_, motor_vel_target_, motor_kp_target_, motor_kd_target_, motor_tau_target_;
    std::vector<int> close_chain_joint_idx_, motor2urdf_;

    void setup_motors();
    void setup_imu();

    template <typename F>
    void exec_motors_parallel(F&& cmd_func) {
        std::unique_lock<std::mutex> lock(motors_mutex_);
        thread_pool_->run_parallel(motor_bus_offsets_.size() - 1,
            [this, &cmd_func](size_t bus) {
                for (size_t idx = motor_bus_offsets_[bus];
                     idx < motor_bus_offsets_[bus + 1]; ++idx) {
                    cmd_func(motors_[idx], static_cast<int>(idx));
                }
            });
    }
    void throw_if_motors_offline() const;
    void motors_mit_cmd();
    void forward_close_chain();
};
