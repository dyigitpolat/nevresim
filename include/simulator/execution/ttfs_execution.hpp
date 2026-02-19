#pragma once

#include "simulator/compute_policy/ttfs_quantized_compute.hpp"

#include <cstddef>

namespace nevresim {

/// Execution policy for TTFS quantized (cycle-based) mode.
///
/// Structurally mirrors ``SpikingExecution``: the **outer loop** iterates
/// over ``(Latency + 1) * S`` discrete cycles.  On each cycle the compute
/// policy processes every core whose latency-gated window includes that
/// cycle (a core with hop-latency L is active during cycles
/// ``[L*S, (L+1)*S)``).
///
/// Neurons are **stateful** — Phase 1 (initial charge) runs on the first
/// active cycle and Phase 2 (fire-once check + constant ramp) advances
/// one step per subsequent active cycle.
///
/// After all cycles complete the output buffer is read once (output
/// neurons hold their final activation values).
///
template <int S, int Latency>
class TTFSExecution
{
public:
    using compute_policy_t = TTFSQuantizedCompute<S>;

    template <typename ConcreteComputePolicy>
    constexpr static auto execute(
        const auto& input,
        auto& chip)
    {
        constexpr int total_cycles = (Latency + 1) * S;

        for (int cycle = 0; cycle < total_cycles; ++cycle) {
            ConcreteComputePolicy::compute(chip, input, cycle);
        }

        return ConcreteComputePolicy::read_output_buffer(chip, input);
    }
};

} // namespace nevresim
