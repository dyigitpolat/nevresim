#pragma once

#include <array>
#include <algorithm>
#include <random>
#include <cstddef>

#include "constants.hpp"
#include "types.hpp"
#include "input_loader.hpp"

namespace nevresim {

template <
    std::size_t InputSize,
    std::size_t SpikeTrainLength>
class SpikeSource
{
    using spike_source_t = std::array<spike_t, InputSize>;

    static spike_source_t generate_spikes(
        const InputLoader& loader)
    {
        static std::random_device device;
        static std::mt19937 engine(device());

        spike_source_t source{};
        std::ranges::generate(loader.input_, source, 
            [&](auto item)
            {
                return static_cast<spike_t>(
                    std::discrete_distribution({1.0-item, item})(engine));
            });
        
        return source;
    }
};

} // namespace nevresim