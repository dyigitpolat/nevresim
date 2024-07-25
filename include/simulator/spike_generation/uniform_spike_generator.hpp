#pragma once

#include "common/constants.hpp"
#include "common/types.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace nevresim {
template <std::size_t InputSize, int SimulationLength>
class UniformSpikeGenerator
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
            [cycle](auto item)
            {
                int N = std::llround(item * SimulationLength);
                
                if (N == 0) return static_cast<spike_t>(0);
                if (N == SimulationLength) return static_cast<spike_t>(1);
                if (cycle >= SimulationLength) return static_cast<spike_t>(0);
                
                double spacing = static_cast<double>(SimulationLength) / N;
                return static_cast<spike_t>(
                    (std::floor(cycle / spacing) < N) && 
                    (std::floor(std::fmod(cycle, spacing)) == 0)
                );
            });
        
        return source;
    }
};

} // namespace nevresim