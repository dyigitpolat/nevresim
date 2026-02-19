#pragma once

#include "simulator/compute_policy/compute_policy_base.hpp"

namespace nevresim 
{

/// Compute policy for TTFS quantized (cycle-based) mode.
///
/// Each neuron simulates S discrete time steps per the B1-model TTFS
/// dynamics.  The neuron is **stateful**: Phase 1 (initial charge) runs
/// on the first active cycle, and Phase 2 (fire-once check + ramp)
/// advances one step per subsequent cycle.
///
/// Latency gating is **enabled** with ``LatencyScale = S``, so a core
/// with hop-latency L becomes active at cycle ``L * S``.  This matches
/// the Python-side ``for cycle in range(total_cycles)`` outer loop.
///
/// Template parameter ``S`` is the number of discrete time steps.
///
template <int S>
class TTFSQuantizedCompute
{
public:
    using signal_t = double;
    static constexpr int time_steps = S;

    template <typename Chip>
    using concrete_policy_t =
        ComputePolicyBase<Chip, signal_t, true, true, S>;
};

} // namespace nevresim
