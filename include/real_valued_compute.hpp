#pragma once

#include "compute_policy_base.hpp"
namespace nevresim 
{

class RealValuedComputeBase
{
public: 
    using signal_t = MembranePotential<weight_t>;
    using base_t = RealValuedComputeBase;
};

template <typename Chip>
class RealValuedCompute : 
    public RealValuedComputeBase, 
    public ComputePolicyBase<Chip, RealValuedComputeBase::signal_t>
{};

} // namespace nevresim