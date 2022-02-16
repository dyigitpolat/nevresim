#pragma once

#include "types.hpp"

namespace nevresim {
    
template <
    std::size_t AxonCount,
    std::size_t NeuronCount,
    threshold_t Threshold>
class Core
{
    using neuron_t = Neuron<AxonCount, Threshold>;
    using neurons_array_t = std::array<neuron_t, NeuronCount>;
    using output_array_t = std::array<spike_t, NeuronCount>;

    neurons_array_t neurons_{};
    output_array_t output_spikes_{};

public:    
    constexpr 
    Core(neurons_array_t neurons) : neurons_(neurons) {}

    constexpr
    const output_array_t& get_output_spikes() const & 
    {
        return output_spikes_;
    }

    constexpr
    void compute(const auto& incoming_spikes)
    {
        std::ranges::transform(
            neurons_, std::ranges::begin(output_spikes_),
            [&](auto& neuron) { return neuron.compute(incoming_spikes); }
        );
    }
};

} // namespace nevresim