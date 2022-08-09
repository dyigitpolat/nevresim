#pragma once

#include "chip_weights.hpp"
#include "types.hpp"
#include "neuron.hpp"

#include <array>
#include <ranges>
#include <algorithm>
#include <cstddef>
#include <istream>

namespace nevresim {

template <
    std::size_t AxonCount,
    std::size_t NeuronCount,
    MembraneLeak<weight_t> LeakAmount = 0>
class Core
{
    using neuron_t = Neuron<AxonCount, LeakAmount>;
    using neurons_array_t = std::array<neuron_t, NeuronCount>;
    using output_array_t = std::array<spike_t, NeuronCount>;
    using real_output_array_t = 
        std::array<MembranePotential<weight_t>, NeuronCount>;

    neurons_array_t neurons_{};
    output_array_t output_spikes_{};
    real_output_array_t output_signals_{};

public:
    constexpr 
    Core() : neurons_{} {}

    constexpr 
    Core(neurons_array_t neurons) : neurons_(neurons) {}

    constexpr
    const output_array_t& get_output_spikes() const & 
    {
        return output_spikes_;
    }

    constexpr
    const real_output_array_t& get_output_signals() const & 
    {
        return output_signals_;
    }

    constexpr
    void compute(const auto& incoming_spikes)
    {
        std::ranges::transform(
            neurons_, std::ranges::begin(output_spikes_),
            [&](auto& neuron) { return neuron.compute(incoming_spikes); }
        );
    }

    constexpr 
    void compute_real(const auto& incoming_signals)
    {
        std::ranges::transform(
            neurons_, std::ranges::begin(output_signals_),
            [&](auto& neuron) { return neuron.compute_real(incoming_signals); }
        );
    }

    constexpr
    void reset()
    {
        for(auto& neuron : neurons_)
        {
            neuron.reset();
        }
    }

    constexpr
    void load_weights(const CoreWeights<NeuronCount, AxonCount>& weights)
    {
        for(
            auto neuron_iter = std::begin(weights.neurons_); 
            auto& neuron : neurons_)
        {
            neuron.load_weights(*neuron_iter++);
        }
    }
};

} // namespace nevresim