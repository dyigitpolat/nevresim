#pragma once

#include "simulator/compute_policy/spiking_compute.hpp"
#include "simulator/compute_policy/real_valued_compute.hpp"

#include <numeric>

namespace nevresim {

template <typename ComputePolicy>
struct NeuronCompute;

template <>
struct NeuronCompute<SpikingComputeBase>
{
    static constexpr SpikingComputeBase::signal_t
    compute(auto& neuron, const auto& incoming_signal)
    {
        neuron.leaky_integrate(incoming_signal);
        return neuron.fire();
    }
};

template <>
struct NeuronCompute<RealValuedComputeBase>
{
    static constexpr RealValuedComputeBase::signal_t
    compute(auto& neuron, const auto& incoming_signal)
    {
        return std::max(
            static_cast<RealValuedComputeBase::signal_t>(0), 
            std::inner_product(
                std::begin(neuron.weights_), std::end(neuron.weights_),
                std::begin(incoming_signal), 
                static_cast<RealValuedComputeBase::signal_t>(0)
            ) + neuron.bias_);
    }
};

} // namespace nevresim