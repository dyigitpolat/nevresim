#pragma once

#include "simulator/compute_policy/compute_policy_base.hpp"
#include "simulator/compute_policy/fire_policy/default_fire.hpp"

namespace nevresim 
{

template <typename FirePolicy = DefaultFirePolicy>
class SpikingCompute
{
public: 
    using signal_t = spike_t;
    using fire_policy_t = FirePolicy;

    template <typename Chip>
    using concrete_policy_t = 
        ComputePolicyBase<Chip, signal_t, FirePolicy::always_on_every_cycle, FirePolicy::latency_gated>;
};

} // namespace nevresim
