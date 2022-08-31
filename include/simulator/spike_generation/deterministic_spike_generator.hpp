#pragma once

#include "common/constants.hpp"
#include "common/types.hpp"

#include <algorithm>

namespace nevresim {

template <std::size_t InputSize, raw_input_t Threshold = 0.5>
class DeterministicSpikeGenerator
{
public:
    using spike_source_t = std::array<spike_t, InputSize>;

    constexpr
    static spike_source_t generate_spikes(
        const auto& input)
    {
        spike_source_t source{};
        std::transform(
            std::cbegin(input), std::cend(input), 
            std::begin(source), 
            [&](auto item)
            {
                return static_cast<spike_t>(
                    item > Threshold
                );
            });
        
        return source;
    }
};

} // namespace nevresim