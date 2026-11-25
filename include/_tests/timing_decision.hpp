#pragma once

#include "simulator/chip/chip.hpp"
#include "simulator/compute_policy/compare_policy/inclusive_compare.hpp"
#include "simulator/compute_policy/compare_policy/strict_compare.hpp"
#include "simulator/compute_policy/timing_decision.hpp"
#include "simulator/compute_policy/ttfs_quantized_compute.hpp"
#include "simulator/execution/real_valued_execution.hpp"
#include "simulator/execution/ttfs_execution.hpp"
#include "simulator/recording/output_format.hpp"

#include "test_util.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace nevresim {
namespace tests {

// The rule the integer decision stands for, by search: the first step k in
// [0, s) at which m / (2s) + k theta / s reaches theta (inclusive) or passes
// it (strict), -1 when no step of the window does.
constexpr int search_fire_step(
    std::int64_t m, std::int64_t theta, int s, bool strict)
{
    const std::int64_t target = 2 * static_cast<std::int64_t>(s) * theta;
    for (int k = 0; k < s; ++k)
    {
        const std::int64_t reached = m + 2 * k * theta;
        if (strict ? (reached > target) : (reached >= target))
            return k;
    }
    return -1;
}

constexpr bool test_timing_decision_rounding()
{
    namespace detail = timing::detail;

    // Round half to even, on both signs.
    const bool rint_ok =
        detail::rint(0.5) == 0.0 && detail::rint(1.5) == 2.0
        && detail::rint(2.5) == 2.0 && detail::rint(-0.5) == 0.0
        && detail::rint(-1.5) == -2.0 && detail::rint(-2.5) == -2.0
        && detail::rint(3.2) == 3.0 && detail::rint(-3.7) == -4.0
        && detail::rint_fast(2.5) == 2.0 && detail::rint_fast(-1.5) == -2.0;

    // Floor and ceiling division with negative operands.
    const bool div_ok =
        detail::floor_div(7, 2) == 3 && detail::floor_div(-7, 2) == -4
        && detail::floor_div(7, -2) == -4 && detail::floor_div(-6, 2) == -3
        && detail::ceil_div(7, 2) == 4 && detail::ceil_div(-7, 2) == -3
        && detail::ceil_div(-6, 2) == -3 && detail::ceil_div(-1, 3) == 0;

    return rint_ok && div_ok;
}

constexpr bool test_timing_decision_fire_step()
{
    // The closed-form integer decision equals the search on every membrane,
    // threshold and window of a small sweep, for both comparators.
    constexpr std::array<std::int64_t, 4> thetas{1, 2, 3, 5};
    constexpr std::array<int, 5> windows{1, 2, 4, 7, 10};
    for (const auto theta : thetas)
        for (const auto s : windows)
            for (std::int64_t m = -60; m <= 60; ++m)
                for (const bool strict : {false, true})
                    if (timing::lattice_fire_step(m, theta, s, strict)
                            != search_fire_step(m, theta, s, strict))
                        return false;

    // V = 1/2, theta 1, S 4 meets the threshold at step 2: the inclusive
    // comparator fires there, the strict one a step later.
    const bool tie_ok =
        timing::lattice_fire_step(4, 1, 4, false) == 2
        && timing::lattice_fire_step(4, 1, 4, true) == 3
        && timing::ramp_fire_step(0.5, 1.0, 4, false) == 2
        && timing::ramp_fire_step(0.5, 1.0, 4, true) == 3;

    // 0.3 + 7 * 0.1 is 0.9999999999999999 in floating point: the ramp passes
    // the tie at step 7 and fires at step 8, the integer decision fires at 7.
    const bool exact_ok =
        timing::ramp_fire_step(0.3, 1.0, 10, false) == 8
        && timing::lattice_fire_step(6, 1, 10, false) == 7;

    const bool level_ok =
        timing::level(4, 0) == 1.0 && timing::level(4, 3) == 0.25
        && timing::level(4, -1) == 0.0;

    return tie_ok && exact_ok && level_ok;
}

constexpr bool test_timing_decision_membrane()
{
    // Integral weights, bias and threshold with inputs on the 1 / (2S) grid
    // are on the lattice: m = 2S (3 * 0.125 - 2 * 0.5 + 5 * 1 - 0.875 + 0.25
    // + 4 * 0.75 - 6 * 0.0625 - 3) = 16 * 3.375 = 54. The off-grid input 0.33
    // is read with weight 0 and does not count.
    constexpr int s{8};
    constexpr std::array<int, 9> weights{3, -2, 5, 0, 7, -1, 1, 4, -6};
    constexpr std::array<double, 9> inputs{
        0.125, 0.5, 1.0, 0.33, 0.0, 0.875, 0.25, 0.75, 0.0625};
    constexpr auto neuron = timing::lattice_neuron(weights, -3.0, 5.0, s);
    constexpr auto membrane =
        timing::lattice_membrane(neuron, weights, inputs, s);
    constexpr auto reference =
        timing::detail::lattice_membrane_any(neuron, weights, inputs, s);
    const bool lattice_ok =
        neuron.integral && membrane.lattice && membrane.m == 54
        && reference.lattice && reference.m == membrane.m;

    // A fractional bias, a threshold below 1 or an off-grid input read with
    // a nonzero weight leaves the lattice.
    constexpr std::array<int, 2> pair_weights{1, 1};
    constexpr std::array<double, 2> off_grid{0.5, 0.33};
    const bool off_ok =
        !timing::lattice_neuron(weights, 0.5, 5.0, s).integral
        && !timing::lattice_neuron(weights, -3.0, 0.5, s).integral
        && !timing::lattice_membrane(
            timing::lattice_neuron(pair_weights, 0.0, 1.0, s),
            pair_weights, off_grid, s).lattice;

    return lattice_ok && off_ok;
}

// One quantized TTFS neuron on an exact tie: weight 1, threshold 1, no bias,
// input 0.3 and S = 10. The membrane V = 0.3 meets the threshold at step 7
// (0.3 + 7 / 10 = 1), so the inclusive comparator latches (10 - 7) / 10 and
// the strict one, a step later, (10 - 8) / 10. A float ramp reads the tie as
// 0.9999999999999999 and fires the inclusive neuron at step 8 as well.
template <typename Con, typename Span, std::size_t core_count, std::size_t in_id>
consteval auto generate_connections_timing_tie()
{
    std::array<Con, core_count> cons;

    cons[0].spans_[0] = Span{in_id, 0, 1};
    cons[0].span_count_ = 1;

    return cons;
}

template <typename Src, std::size_t output_size>
consteval auto generate_outputs_timing_tie()
{
    std::array<Src, output_size> outs;

    outs[0] = Src{0, 0};

    return outs;
}

template <typename ComparePolicy>
constexpr double run_quantized_ttfs_tie()
{
    using weight_t = double;
    constexpr std::size_t axon_count{1};
    constexpr std::size_t neuron_count{1};
    constexpr std::size_t core_count{1};
    constexpr std::size_t input_size{1};
    constexpr std::size_t output_size{1};
    constexpr std::size_t max_spans{1};
    constexpr MembraneLeak<weight_t> leak{0};
    constexpr int S{10};
    constexpr int latency{0};

    using Src = SpikeSource;
    using Span = SourceSpan;
    using Con = CoreSpanConnection<max_spans>;

    auto chip = generate_test_chip<
        weight_t,
        generate_connections_timing_tie<Con, Span, core_count, in>(),
        generate_outputs_timing_tie<Src, output_size>(),
        axon_count,
        neuron_count,
        core_count,
        input_size,
        output_size,
        max_spans,
        leak,
        TTFSQuantizedCompute<S, ComparePolicy>> ();

    using ChipW = ChipWeights<chip.config_, weight_t, weight_t>;
    using CoreW = CoreWeights<chip.config_, weight_t, weight_t>;
    using NeurW = NeuronWeights<chip.config_, weight_t, weight_t>;
    using Ws = std::array<Weight<weight_t>, axon_count>;

    ChipW weights{{
        CoreW{{NeurW{Ws{1.0}, 1.0}}, latency}
    }};
    chip.load_weights(weights);

    std::array<raw_input_t, input_size> input{0.3};

    auto buffer = chip.template execute<
        TTFSExecution<S, latency, ComparePolicy>>(input);

    chip.reset();
    return buffer[0];
}

constexpr bool test_quantized_ttfs_fires_on_exact_tie()
{
    return run_quantized_ttfs_tie<InclusiveCompare>() == 3.0 / 10
        && run_quantized_ttfs_tie<StrictCompare>() == 2.0 / 10;
}

// The quantized timing execution writes every digit of a double, every other
// execution keeps the stream's default.
static_assert(
    output_digits<TTFSExecution<10, 0, InclusiveCompare>>::value
        == std::numeric_limits<double>::max_digits10);
static_assert(output_digits<RealValuedExecution>::value == 0);

} // namespace nevresim::tests
} // namespace nevresim
