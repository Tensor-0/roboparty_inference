// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#pragma once

#include <algorithm>
#include <cmath>

namespace safety {

// ⭐ 下发目标裁剪（2026-09-22）
//
//   joint_limits 原来只用来【检】实测关节角（obs_manager.cpp:232），
//   真正下发给电机的目标全程不裁：clip_actions = 18.0 ≈ 不裁，
//   再乘 action_scale = 0.25 ⇒ 目标可以离默认角 4.5 rad。
//   策略发散的瞬间，这就不是"关节角"而是一个直接顶到机械挡块的命令。
//
//   裁剪位置固定在【关节坐标系】、进 decouple 之前 ——
//   踝关节解耦在 RobotInterface::apply_action 内部做，那边收到的是关节角。
inline constexpr double kJointLimitMargin = 0.05;  // rad，留在限位内侧的余量

// 把 q 就地裁到 [lo + margin, hi - margin]，返回"是否被改动过"。
//
//   ⚠️ margin 大于半个区间时按半个区间缩。否则 lo + margin > hi - margin，
//      上下界反过来，std::clamp 的前提就没了。
//   ⚠️ lo > hi（配置里写反了）按 [hi, lo] 处理并返回 true，让调用方报出来 ——
//      静默把配置错误吞掉比裁剪本身更危险。
//   ⚠️ q 非有限（NaN / ±inf）时【不动它】并返回 false。对 NaN 来说"裁到一个数"
//      没有意义（NaN 与任何数比较都是 false，clamp 会原样放行），
//      非有限值在调用方单独处理：保持上一帧目标 + 切 PD。
inline bool clip_joint_target(double& q, double lo, double hi, double margin) {
    if (!std::isfinite(q)) {
        return false;
    }
    if (hi < lo) {
        std::swap(lo, hi);
    }
    const double half = 0.5 * (hi - lo);
    const double m = std::min(std::abs(margin), half);
    const double clipped = std::clamp(q, lo + m, hi - m);
    const bool changed = (clipped != q);
    q = clipped;
    return changed;
}

}  // namespace safety
