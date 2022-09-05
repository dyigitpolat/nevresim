#pragma once

#include "simulator/compute_policy/compute_policy_base.hpp"

namespace nevresim 
{

class SpikingCompute
{
public: 
    using signal_t = spike_t;
    using base_t = SpikingCompute;

};

template <typename Chip>
class ConcreteSpikingCompute :
    public SpikingCompute, 
    public ComputePolicyBase<Chip, SpikingCompute::signal_t>
{};

} // namespace nevresim