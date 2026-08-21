#pragma once

#include "simulator/compute_policy/compute_policy_base.hpp"
#include "simulator/compute_policy/fire_policy/default_fire.hpp"
// Both integration policies ride the compute policy's own header, so a
// generated main.cpp that names one needs NO new #include line and the emitted
// text of every default-point program stays byte-identical.
#include "simulator/compute_policy/integration_policy/event_serial_integrate.hpp"
#include "simulator/compute_policy/integration_policy/whole_vector_integrate.hpp"

namespace nevresim
{

template <
    typename FirePolicy = DefaultFirePolicy,
    typename IntegrationPolicy = WholeVectorIntegrate>
class SpikingCompute
{
public:
    using signal_t = spike_t;
    using fire_policy_t = FirePolicy;
    using integration_policy_t = IntegrationPolicy;

    template <typename Chip>
    using concrete_policy_t =
        ComputePolicyBase<Chip, signal_t, typename FirePolicy::always_on_policy, FirePolicy::latency_gated>;
};

} // namespace nevresim
