#pragma once

#include "common/constants.hpp"
#include "common/types.hpp"
#include "loaders/input_loader.hpp"

#include <array>
#include <algorithm>
#include <random>
#include <cstddef>

namespace nevresim {

template <std::size_t InputSize>
class SpikeGenerator
{
public:
    using spike_source_t = std::array<spike_t, InputSize>;

    static spike_source_t generate_spikes(
        const auto& input)
    {
        static std::random_device device;
        static std::mt19937 engine(device());
        
        return generate_spikes(input, engine);
    }

    constexpr
    static spike_source_t generate_spikes(
        const auto& input, auto& random_engine)
    {
        spike_source_t source{};
        std::transform(
            std::cbegin(input), std::cend(input), 
            std::begin(source), 
            [&](auto item)
            {
                return static_cast<spike_t>(
                    std::discrete_distribution({1.0-item, item})
                    (random_engine)
                );
            });
        
        return source;
    }
};

} // namespace nevresim