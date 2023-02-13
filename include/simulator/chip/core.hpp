#pragma once

#include "simulator/chip_weights.hpp"
#include "common/types.hpp"
#include "simulator/chip/neuron.hpp"

#include <array>
#include <ranges>
#include <algorithm>
#include <cstddef>
#include <istream>

namespace nevresim {

template <
    typename Config,
    typename ComputePolicy
    >
class Core
{
    using signal_t = typename ComputePolicy::signal_t;
    using neuron_t = Neuron<Config, ComputePolicy>;
    using neurons_array_t = std::array<neuron_t, Config::neuron_count_>;
    using output_array_t = std::array<signal_t, Config::neuron_count_>;

    neurons_array_t neurons_{};
    output_array_t output_{};

public:
    constexpr 
    Core() : neurons_{} {}

    constexpr 
    Core(neurons_array_t neurons) : neurons_(neurons) {}

    constexpr
    const output_array_t& get_output() const & 
    {
        return output_;
    }

    constexpr void compute(const auto& incoming_signals)
    {
        std::ranges::transform(
            neurons_, std::ranges::begin(output_),
            [&](auto& neuron) { 
                return neuron.compute(incoming_signals); 
            }
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
    void load_weights(
        const CoreWeights<Config{}, typename Config::weight_t>& weights)
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