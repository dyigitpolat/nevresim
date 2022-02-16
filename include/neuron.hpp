#pragma once

#include "types.hpp"

namespace nevresim {
template <
    std::size_t AxonCount,
    threshold_t Threshold,
    membrane_leak_t LeakAmount = 0>
class Neuron
{
    using weights_array_t = std::array< weight_t, AxonCount>;

    weights_array_t weights_{};
    membrane_potential_t membrane_potential_{};

    constexpr void leaky_integrate(const auto& incoming_spikes)
    {
        membrane_potential_ += std::inner_product(
            std::begin(weights_), std::end(weights_),
            std::begin(incoming_spikes), 0
        ) - LeakAmount;
    }

    constexpr spike_t fire()
    {
        spike_t spike = Threshold < membrane_potential_;

        if(spike)
        {
            membrane_potential_ -= Threshold;
        }
        
        return spike;
    }

public:
    constexpr 
    Neuron(weights_array_t weights) : weights_{weights} {}

    constexpr
    spike_t compute(const auto& incoming_spikes)
    {
        leaky_integrate(incoming_spikes);
        return fire();
    }
};

} // namespace nevresim