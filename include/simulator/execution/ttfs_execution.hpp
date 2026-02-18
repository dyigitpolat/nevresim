#pragma once

#include "simulator/compute_policy/ttfs_quantized_compute.hpp"

#include <cstddef>

namespace nevresim {

/// Execution policy for TTFS quantized (cycle-based) mode.
///
/// Structurally identical to ``TTFSContinuousExecution``: a single forward
/// sweep through all cores, iterating ``core_count_`` times so that
/// signals propagate through the entire network (each iteration
/// propagates activations one hop further).
///
/// The only difference from the continuous path is the compute policy:
/// ``TTFSQuantizedCompute<S>`` makes each neuron internally step through
/// ``S`` discrete time steps (Phase 1 + Phase 2 of the B1-model),
/// producing quantised activations in {0, 1/S, 2/S, …, 1}.
///
template <int S>
class TTFSExecution
{
public:
    using compute_policy_t = TTFSQuantizedCompute<S>;

    template <typename ConcreteComputePolicy>
    constexpr static auto execute(
        const auto& input,
        auto& chip)
    {
        for(std::size_t i = 0; i < chip.config_.core_count_; ++i){
            ConcreteComputePolicy::compute(chip, input, i);
        }

        return ConcreteComputePolicy::read_output_buffer(chip, input);
    }
};

} // namespace nevresim
