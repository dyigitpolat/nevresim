#pragma once

#include "_tests/test_util.hpp"
#include "_tests/single_core.hpp"
#include "_tests/multi_core.hpp"
#include "_tests/real_valued_single_core.hpp"
#include "_tests/real_valued_multi_core.hpp"
#include "_tests/xor_test/xor_test.hpp"
#include "_tests/spike_train_generator.hpp"
#include "_tests/stage_input_clock.hpp"
#include "_tests/timing_decision.hpp"

#include "simulator/compute_policy/fire_policy/novena_fire.hpp"

#include <iostream>

namespace nevresim
{
namespace tests
{

static_assert(test_spike_train_generator_replays_flat_train());
static_assert(test_single_core());
static_assert(test_3_core_2x2());
static_assert(test_single_real_valued_core());
static_assert(test_multi_real_valued_core());
static_assert(test_stage_input_read_in_core_local_time());
static_assert(test_timing_decision_rounding());
static_assert(test_timing_decision_fire_step());
static_assert(test_timing_decision_membrane());
static_assert(test_quantized_ttfs_fires_on_exact_tie());

inline void run_all()
{
    report_result("single core", test_single_core);
    report_result("3 core 2x2", test_3_core_2x2);
    report_result("real valued single core", test_single_real_valued_core);
    report_result("real valued multi core", test_multi_real_valued_core);
    report_result("stage input in core-local time", test_stage_input_read_in_core_local_time);
    report_result("timing decision rounding", test_timing_decision_rounding);
    report_result("timing decision fire step", test_timing_decision_fire_step);
    report_result("timing decision membrane", test_timing_decision_membrane);
    report_result("quantized ttfs on an exact tie", test_quantized_ttfs_fires_on_exact_tie);
    report_result("xor test", test_xor<>);
    report_result("xor test novena fire", test_xor<NovenaFirePolicy>);
}

} // namespace nevresim::tests
} // namespace nevresim