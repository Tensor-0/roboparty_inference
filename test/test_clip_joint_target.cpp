// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi
//
// safety_clip.hpp 的本地用例。**不依赖 ROS / Eigen / onnxruntime**，
// 就是为了能在开发机上直接编、直接跑（本机没有板子的那套环境）：
//
//   g++ -std=c++17 -O1 -Wall -o /tmp/test_clip test/test_clip_joint_target.cpp && /tmp/test_clip
//
// 覆盖：正常裁剪、边界、窄区间、配置写反、非有限值，
// 以及 dm10 真实限位下"PD 目标不会被裁掉"这条不变量。

#include "../src/utils/safety_clip.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

static int g_failures = 0;

static void check(bool ok, const char* what) {
    if (!ok) {
        ++g_failures;
        std::printf("  FAIL: %s\n", what);
    }
}

static void check_double(double got, double want, const char* what) {
    if (!(std::abs(got - want) < 1e-12)) {
        ++g_failures;
        std::printf("  FAIL: %s (got %.15g, want %.15g)\n", what, got, want);
    }
}

// 关节限位 [lo, hi] + 留 0.05 余量
static constexpr double kMargin = safety::kJointLimitMargin;

int main() {
    std::printf("clip_joint_target\n");

    // ── 区间内：不动，报 false ────────────────────────────────
    {
        double q = 0.3;
        const bool changed = safety::clip_joint_target(q, -1.0, 1.0, kMargin);
        check(!changed, "in-range value must not be reported as changed");
        check_double(q, 0.3, "in-range value must not move");
    }
    // ── 上界：裁到 hi - margin ────────────────────────────────
    {
        double q = 12.0;
        const bool changed = safety::clip_joint_target(q, -1.0, 1.0, kMargin);
        check(changed, "above hi must be reported as changed");
        check_double(q, 1.0 - kMargin, "above hi must clamp to hi - margin");
    }
    // ── 下界：裁到 lo + margin ────────────────────────────────
    {
        double q = -12.0;
        const bool changed = safety::clip_joint_target(q, -1.0, 1.0, kMargin);
        check(changed, "below lo must be reported as changed");
        check_double(q, -1.0 + kMargin, "below lo must clamp to lo + margin");
    }
    // ── 边界上恰好是 hi - margin：不动 ────────────────────────
    {
        double q = 0.95;
        const bool changed = safety::clip_joint_target(q, -1.0, 1.0, kMargin);
        check(!changed, "value exactly at hi - margin must not be reported as changed");
        check_double(q, 0.95, "value exactly at hi - margin must not move");
    }
    // ── 窄区间：margin 缩到半个区间，结果落在中点（不能上下界反转）──
    {
        double q = 5.0;
        safety::clip_joint_target(q, -0.02, 0.02, kMargin);
        check_double(q, 0.0, "narrow range must collapse to the midpoint");
    }
    // ── 配置写反 (lo > hi)：按 [hi, lo] 处理 ──────────────────
    //   ⚠️ 这是纯防御 —— load_config 里已经在启动时把写反的限位抛掉了，
    //      这个分支正常不会走到。写反的语义是"当成正常区间用"，
    //      所以判据是它对不对，不是它报不报 changed。
    {
        double q = 2.0;
        const bool changed = safety::clip_joint_target(q, 1.0, -1.0, kMargin);
        check(changed, "reversed limits: out-of-range value must be clipped");
        check_double(q, 1.0 - kMargin, "reversed limits must clip against the swapped range");
    }
    {
        double q = -0.4;  // 落在换回来的 [-1, 1] 里
        const bool changed = safety::clip_joint_target(q, 1.0, -1.0, kMargin);
        check(!changed, "reversed limits: in-range value must not be clipped");
        check_double(q, -0.4, "reversed limits: in-range value must not move");
    }
    // ── 非有限值：一律不动、报 false（由调用方单独处理）────────
    {
        double q = std::numeric_limits<double>::quiet_NaN();
        const bool changed = safety::clip_joint_target(q, -1.0, 1.0, kMargin);
        check(!changed, "NaN must not be reported as changed");
        check(std::isnan(q), "NaN must be passed through untouched");
    }
    {
        double q = std::numeric_limits<double>::infinity();
        const bool changed = safety::clip_joint_target(q, -1.0, 1.0, kMargin);
        check(!changed, "+inf must not be reported as changed");
        check(std::isinf(q), "+inf must be passed through untouched");
    }

    // ── 不变量：dm10 的 PD 站立目标不能被裁掉 ──────────────────
    //   （joint_default_angle 与 pd_stand_target 在 dm10 上都是全零；
    //     若限位写作导致 0 被裁，PD 保底就永远站不到目标姿态。）
    {
        const std::vector<double> dm10_limits = {
            -1.05, 1.05, -0.52, 0.785, -0.52, 0.785, -0.3, 2.3562, -1.047, 0.3,
            -1.05, 1.05, -0.785, 0.52, -0.785, 0.52, -0.3, 2.3562, -1.047, 0.3};
        for (size_t j = 0; j < dm10_limits.size() / 2; ++j) {
            double q = 0.0;  // dm10 的 PD 站立目标
            const bool changed =
                safety::clip_joint_target(q, dm10_limits[2 * j], dm10_limits[2 * j + 1], kMargin);
            if (changed || q != 0.0) {
                ++g_failures;
                std::printf("  FAIL: dm10 joint %zu: PD target 0.0 got clipped\n", j);
            }
        }
        // 策略发散的极端目标（默认角 + 4.5 rad）必须被裁回限位内
        double q = 0.0 + 4.5;
        check(safety::clip_joint_target(q, -1.05, 1.05, kMargin), "diverged target must be clipped");
        check_double(q, 1.05 - kMargin, "diverged target must land on the hip limit");
    }

    if (g_failures == 0) {
        std::printf("all checks passed\n");
        return 0;
    }
    std::printf("%d check(s) FAILED\n", g_failures);
    return 1;
}
