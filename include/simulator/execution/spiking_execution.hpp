#pragma once

#include "simulator/compute_policy/spiking_compute.hpp"
#include "simulator/compute_policy/fire_policy/default_fire.hpp"

#include <algorithm>
#include <cstddef>

namespace nevresim {

template <
    int SimulationLength,
    std::size_t OutputSize,
    template<auto... Args> class SpikeProvider,
    typename WeightType,
    typename FirePolicy = DefaultFirePolicy>
class SpikingExecution
{
public:
    using compute_policy_t = SpikingCompute<FirePolicy>;

    template <typename ConcreteComputePolicy, typename Chip>
    constexpr static auto execute(
        const auto& input, 
        Chip& chip)
    {
        using weight_t = WeightType;

        std::array<Weight<weight_t>, OutputSize> buffer{};
        for(int i = 0; i < SimulationLength; ++i){
            const auto& spikes{
                SpikeProvider<(Chip::config_).input_size_>
                    ::generate_spikes(input)};

            ConcreteComputePolicy::compute_chip(chip, spikes);

            auto&& out{ConcreteComputePolicy::read_output_buffer(chip, spikes)};
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