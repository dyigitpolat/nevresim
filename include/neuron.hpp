#pragma once

#include <array>
#include <numeric>
#include <cstddef>
#include <istream>

#include "types.hpp"

namespace nevresim {
template <
    std::size_t AxonCount,
    membrane_leak_t LeakAmount = 0>
class Neuron
{
    using weights_array_t = std::array< weight_t, AxonCount>;

    weights_array_t weights_{};
    threshold_t threshold_{};
    membrane_potential_t membrane_potential_{};

    constexpr void leaky_integrate(const auto& incoming_spikes)
    {
        membrane_potential_ += std::inner_product(
            std::begin(weights_), std::end(weights_),
            std::begin(incoming_spikes), static_cast<membrane_potential_t>(0)
        ) - LeakAmount;
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

    constexpr
    spike_t compute(const auto& incoming_spikes)
    {
        leaky_integrate(incoming_spikes);
        return fire();
    }

    void reset()
    {
        membrane_potential_ = 0;
    }

    friend std::istream& operator>>(
        std::istream& weights_stream, Neuron& neuron)
    {
        weights_stream >> neuron.threshold_;
        for(auto& weight : neuron.weights_)
        {
            weights_stream >> weight;
        }

        return weights_stream;
    }
};

} // namespace nevresim