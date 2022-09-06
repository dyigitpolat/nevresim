#pragma once

#include "simulator/compute_policy/spiking_compute.hpp"

#include <algorithm>
#include <cstddef>

namespace nevresim {

template <
    int SimulationLength,
    template<std::size_t InputSize> class SpikeProvider>
class SpikingExecution
{
public:
    using compute_policy_t = SpikingCompute;

    constexpr static auto execute(
        const auto& input, 
        auto& chip,
        const auto& compute_function,
        const auto& output_buffer_read_function)
    {
        std::array<Weight<weight_t>, chip.config_.output_size_> buffer{};
        for(int i = 0; i < SimulationLength; ++i){
            const auto& spikes{
                SpikeProvider<chip.config_.input_size_>
                    ::generate_spikes(input)};

            compute_function(chip, spikes);

            auto&& out = output_buffer_read_function(chip, spikes);
            std::ranges::transform(
                std::ranges::cbegin(out), std::ranges::cend(out),
                std::ranges::cbegin(buffer), std::ranges::cend(buffer),
                std::ranges::begin(buffer), 
                std::plus{});
        }

        return buffer;
    }
};

} // namespace nevresim