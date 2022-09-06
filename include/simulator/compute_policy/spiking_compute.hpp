#pragma once

#include "simulator/compute_policy/compute_policy_base.hpp"

namespace nevresim 
{

class SpikingCompute
{
public: 
    using signal_t = spike_t;

    template <typename Chip>
    using concrete_policy_t = ComputePolicyBase<Chip, signal_t>;
};

} // namespace nevresim