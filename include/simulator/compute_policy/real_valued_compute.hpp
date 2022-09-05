#pragma once

#include "simulator/compute_policy/compute_policy_base.hpp"
namespace nevresim 
{

class RealValuedCompute
{
public: 
    using signal_t = MembranePotential<weight_t>;
    using base_t = RealValuedCompute;
};

template <typename Chip>
class ConcreteRealValuedCompute : 
    public RealValuedCompute, 
    public ComputePolicyBase<Chip, RealValuedCompute::signal_t>
{};

} // namespace nevresim