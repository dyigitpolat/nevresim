#pragma once

#include "simulator/chip_utilities.hpp"

namespace nevresim 
{

template <typename Chip, typename SignalType>
class ComputePolicyBase
{
private:
    consteval
    static auto retrieve_signal_from() 
    {
        return []
            (const Chip& chip, SpikeSource source, const auto& input_buffer) 
            -> SignalType
            {
                if(source.core_ == k_input_buffer_id)
                {
                    return input_buffer[source.neuron_];
                }
                else if(source.core_ == k_no_connection)
                {
                    return SignalType{};
                }
                else
                {
                    return 
                        chip.get_cores()[source.core_]
                            .get_output()[source.neuron_];
                }
            };
    }

    template <std::size_t N> consteval
    static auto retrieve_signals() 
    {
        return []
            (const Chip& chip, auto signal_sources, const auto& input_buffer) {
            std::array<SignalType, N> spikes{};

            [&]<std::size_t ...Idx> (std::index_sequence<Idx...>)
            {
                (
                    (spikes[Idx] 
                        = retrieve_signal_from()(
                            chip, signal_sources[Idx], input_buffer)
                    )
                , ...);
            }(std::make_index_sequence<N>{});

            return spikes;
        };
    }

    consteval
    static auto get_input_for() 
    {
        return []
            (const Chip& chip, core_id_t core_id, const auto& input_buffer){
            return retrieve_signals<Chip::config_.axon_count_>()(
                chip, 
                Chip::mapping_.core_sources_[core_id].sources_,
                input_buffer);
        };
    }

public: 
    consteval
    static auto generate_read_output_buffer()
    {
        return [](const Chip& chip, const auto& input_buffer){
            return 
                retrieve_signals<Chip::config_.output_size_>()(
                    chip, Chip::mapping_.output_sources_, input_buffer);
        };
    }

    consteval 
    static auto generate_compute() 
    {
        return [](Chip& chip, const auto& input_buffer){
            [] <core_id_t ...IDs>(
                Chip& chip_, 
                const auto& input_buffer_, 
                std::index_sequence<IDs...>)
            {
                std::array<
                    std::array<SignalType, Chip::config_.axon_count_>, 
                    Chip::config_.core_count_> axons{};

                ((axons[IDs] = 
                    get_input_for()(chip_, IDs, input_buffer_)), ...);

                (chip_.get_cores()[IDs].compute(axons[IDs]), ...);
                
            } (
                chip, 
                input_buffer,
                std::make_index_sequence<Chip::config_.core_count_>{});
        };
    }
};

} // namespace nevresim