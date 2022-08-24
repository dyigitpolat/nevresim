#pragma once

#include "compute_policy_base.hpp"

namespace nevresim 
{

class SpikingComputeBase
{
public: 
    using signal_t = spike_t;
    using base_t = SpikingComputeBase;

};

template <typename Chip>
class SpikingCompute :
    public SpikingComputeBase, 
    public ComputePolicyBase<Chip, SpikingComputeBase::signal_t>
{};

} // namespace nevresim