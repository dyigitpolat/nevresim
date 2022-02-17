#pragma once

#include <array>
#include <utility>
#include <cstddef>

#include "constants.hpp"
#include "types.hpp"
#include "core.hpp"

namespace nevresim {
struct SpikeSource
{
    core_id_t core_{};
    axon_id_t neuron_{};
};

template<std::size_t AxonCount>
struct CoreConnection {
    std::array<SpikeSource, AxonCount> sources_{};
};

template<
    std::size_t AxonCount,
    std::size_t CoreCount,
    std::size_t OutputSize>
struct ChipConfiguration {
    std::array<CoreConnection<AxonCount>, CoreCount> core_sources_{};
    std::array<SpikeSource, OutputSize> output_sources_{};
};

template <size_t N> consteval
auto make_spike_array(auto iter) 
{
    return 
        [&]<std::size_t ...Idx> (std::index_sequence<Idx...>)
        {
            return std::array<spike_t, N> { ((void)Idx, *(iter++))... };
        }(std::make_index_sequence<N>{});
} 

template<
    std::size_t AxonCount,
    std::size_t NeuronCount,
    std::size_t CoreCount,
    std::size_t InputSize,
    std::size_t OutputSize,
    threshold_t Threshold,
    ChipConfiguration<
        AxonCount,
        CoreCount,
        OutputSize> Configuration
    >
class Chip
{
    using cores_array_t = 
        std::array< Core< AxonCount, NeuronCount, Threshold>, CoreCount>;
    
    using input_buffer_t = 
        std::array< spike_t, InputSize>;

    cores_array_t cores_{};
    input_buffer_t input_buffer_{};

    consteval
    static auto retrieve_spike_from() 
    {
        return [](const Chip& chip, SpikeSource source) -> spike_t {
            if(source.core_ == k_input_buffer_id)
            {
                return chip.input_buffer_[source.neuron_];
            }
            else if(source.core_ == k_no_connection)
            {
                return spike_t{};
            }
            else
            {
                return 
                    chip.cores_[source.core_]
                        .get_output_spikes()[source.neuron_];
            }
        };
    }

    template <std::size_t N> consteval
    static auto retrieve_spikes() 
    {
        return [](const Chip& chip, auto spike_sources) {
            std::array<spike_t, N> spikes{};

            [&]<std::size_t ...Idx> (std::index_sequence<Idx...>)
            {
                (
                    (spikes[Idx] 
                        = retrieve_spike_from()(chip, spike_sources[Idx])
                    )
                , ...);
            }(std::make_index_sequence<N>{});

            return spikes;
        };
    }

    consteval
    static auto get_input_for() 
    {
        return [](const Chip& chip, core_id_t core_id){
            return retrieve_spikes<AxonCount>()(
                chip, Configuration.core_sources_[core_id].sources_);
        };
    }

public:
    constexpr 
    Chip(cores_array_t cores) : cores_(cores) {}
    
    constexpr 
    static auto generate_read_output_buffer()
    {
        return [](const Chip& chip){
            return 
                retrieve_spikes<OutputSize>()(
                    chip, Configuration.output_sources_);
        };
    }

    consteval 
    static auto generate_compute() 
    {
        return [](Chip& chip){
            [] <core_id_t ...IDs>
            (Chip& chip_, std::index_sequence<IDs...>)
            {
                std::array<std::array<spike_t, AxonCount>, CoreCount> 
                    axons{};

                ((axons[IDs] = get_input_for()(chip_, IDs)), ...);
                (chip_.cores_[IDs].compute(axons[IDs]), ...);
                
            } (chip, std::make_index_sequence<CoreCount>{});
        };
    }

    void feed_input_buffer(const input_buffer_t& feed)
    {
        input_buffer_ = feed;
    }
};

} // namespace nevresim