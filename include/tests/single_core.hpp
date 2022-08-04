#pragma once

#include "neuron.hpp"
#include "core.hpp"
#include "chip.hpp"
#include "input_loader.hpp"
#include "weights_loader.hpp"

#include "test_util.hpp"

#include <sstream>

namespace nevresim {
namespace tests {

template <typename Con, typename Src, size_t core_count>
consteval auto generate_connections_single_core_2x2()
{
    std::array<Con, core_count> cons; 

    cons[0].sources_[0] = Src{in,0};
    cons[0].sources_[1] = Src{in,1};

    return cons;
}

template <typename Src, size_t output_size>
consteval auto generate_outputs_single_core_2x2()
{
    std::array<Src, output_size> outs;    

    outs[0] = Src{0,0};
    outs[1] = Src{0,1};

    return outs;
}

bool test_single_core()
{
    constexpr std::size_t axon_count{2};
    constexpr std::size_t neuron_count{2};
    constexpr std::size_t core_count{1};
    constexpr std::size_t input_size{2};
    constexpr std::size_t output_size{2};
    constexpr membrane_leak_t leak{0};

    using Src = SpikeSource;
    using Con = CoreConnection<axon_count>;

    constinit static auto chip = generate_test_chip<
        generate_connections_single_core_2x2<Con, Src, core_count>(),
        generate_outputs_single_core_2x2<Src, output_size>(),
        axon_count,
        neuron_count,
        core_count,
        input_size,
        output_size,
        leak> ();
    constexpr auto compute = chip.generate_compute();
    constexpr auto read_output_buffer = chip.generate_read_output_buffer();

    WeightsLoader<core_count, neuron_count, axon_count> weights_loader{};
    std::stringstream weights_stream("0 3 -2 3 7 -5");
    if(weights_stream)
    {
        weights_stream >> weights_loader;
    }
    chip.load_weights(weights_loader.chip_weights_);

    std::array<raw_input_t, 2> input{1.0, 1.0};
        
    auto buffer = 
        nevresim::ChipExecutor<nevresim::SpikingExecution<4>>::execute(
            input, chip, compute, read_output_buffer
        );

    chip.reset();
    return buffer == std::array<weight_t, 2>{4, 2};
}

} // namespace nevresim::tests
} // namespace nevresim