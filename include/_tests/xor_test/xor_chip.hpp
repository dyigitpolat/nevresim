
#pragma once

#include "simulator/chip/neuron.hpp"
#include "simulator/chip/core.hpp"
#include "simulator/chip/chip.hpp"

namespace nevresim::tests
{

template <typename Con, typename Src, size_t core_count, size_t in, size_t off>
consteval auto generate_xor_chip_connections()
{
    std::array<Con, core_count> cons; 
        
    cons[0].sources_[0] = Src{in,0};
    cons[0].sources_[1] = Src{in,1};
    cons[0].sources_[2] = Src{off,0};
    cons[1].sources_[0] = Src{0,0};
    cons[1].sources_[1] = Src{0,1};
    cons[1].sources_[2] = Src{0,2};
    cons[2].sources_[0] = Src{1,0};
    cons[2].sources_[1] = Src{1,1};
    cons[2].sources_[2] = Src{1,2};

    return cons;
}

template <typename Src, size_t output_size>
consteval auto generate_xor_chip_outputs()
{
    std::array<Src, output_size> outs;
    
    outs[0] = Src{2,0};
    outs[1] = Src{2,1};

    return outs;
}

template <typename ComputePolicy>
consteval auto generate_xor_chip()
{
    constexpr std::size_t axon_count{3};
    constexpr std::size_t neuron_count{3};
    constexpr std::size_t core_count{3};
    constexpr std::size_t input_size{2};
    constexpr std::size_t output_size{2};
    constexpr MembraneLeak<weight_t> leak{0};

    using Cfg = ChipConfiguration<
        axon_count,
        neuron_count,
        core_count,
        input_size,
        output_size,
        leak
    >;

    using Map = Mapping<Cfg>;
    using Src = SpikeSource;
    using Con = CoreConnection<axon_count>;

    constexpr core_id_t in = k_input_buffer_id;
    constexpr core_id_t off = k_no_connection;

    using Chip = Chip<
        Cfg, 
        Map{
            generate_xor_chip_connections<Con, Src, core_count, in, off>(),
            generate_xor_chip_outputs<Src, output_size>()}, 
        ComputePolicy>;

    constexpr Chip chip{};
    return chip;
}

} // namespace nevresim::tests