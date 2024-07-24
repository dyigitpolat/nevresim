#pragma once

#include "common/constants.hpp"
#include "common/types.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace nevresim {
template <std::size_t InputSize, int SimulationLength>
class FrontLoadedSpikeGenerator
{
public:
    using spike_source_t = std::array<spike_t, InputSize>;

    constexpr
    static spike_source_t generate_spikes(
        const auto& input, int cycle)
    {
        spike_source_t source{};
        std::transform(
            std::cbegin(input), std::cend(input), 
            std::begin(source), 
            [&](auto item)
            {
                return static_cast<spike_t>(
                    std::llround(item * SimulationLength) > cycle
                );
            });
        
        return source;
    }
};

} // namespace nevresim