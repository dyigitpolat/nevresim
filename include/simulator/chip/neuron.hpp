#pragma once

#include "simulator/chip_weights.hpp"
#include "common/types.hpp"
#include "simulator/compute_policy/neuron_compute.hpp"

#include <array>
#include <numeric>
#include <cstddef>
#include <istream>
#include <type_traits>

namespace nevresim {

template <typename Config>
class Neuron
{
    using weights_array_t = std::array< Weight<weight_t>, Config::axon_count_>;

    weights_array_t weights_{};
    Threshold<weight_t> threshold_{};
    Bias<weight_t> bias_{};
    MembranePotential<weight_t> membrane_potential_{};

    constexpr void leaky_integrate(const auto& incoming_spikes)
    {
        membrane_potential_ += 
            std::inner_product(
                std::begin(weights_), std::end(weights_),
                std::begin(incoming_spikes), 
                static_cast<MembranePotential<weight_t>>(0))
            + bias_ 
            - Config::leak_amount_;
    }

    constexpr spike_t fire()
    {
        spike_t spike = threshold_ < membrane_potential_;

        if(spike)
        {
            membrane_potential_ -= threshold_;
        }
        
        return spike;
    }

public:
    constexpr 
    Neuron() : weights_{} {}

    constexpr 
    Neuron(weights_array_t weights) : weights_{weights} {}

    template <typename ComputePolicy> constexpr ComputePolicy::signal_t 
    compute(const auto& incoming_signal)
    {
        return NeuronCompute<ComputePolicy>::compute(*this, incoming_signal);
    }

    constexpr
    void reset()
    {
        membrane_potential_ = 0;
    }

    constexpr
    void load_weights(const NeuronWeights<Config{}>& weights)
    {
        threshold_ = weights.threshold_;
        weights_ = weights.weights_;
        bias_ = weights.bias_;
    }

    template <typename T>
    friend class NeuronCompute;
};

} // namespace nevresim