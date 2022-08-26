#pragma once

#include "neuron.hpp"
#include "core.hpp"
#include "chip.hpp"
#include "input_loader.hpp"
#include "weights_loader.hpp"
#include "deterministic_spike_generator.hpp"
#include "spiking_compute.hpp"

#include "test_util.hpp"

#include <sstream>

namespace nevresim {
namespace tests {

template <typename Con, typename Src, size_t core_count>
consteval auto generate_connections_3_cores_2x2()
{
    std::array<Con, core_count> cons; 

    cons[0].sources_[0] = Src{in,0};
    cons[0].sources_[1] = Src{in,1};

    cons[1].sources_[0] = Src{0,1};
    cons[1].sources_[1] = Src{off,0};

    cons[2].sources_[0] = Src{off,0};
    cons[2].sources_[1] = Src{0,0};

    return cons;
}

template <typename Src, size_t output_size>
consteval auto generate_outputs_3_cores_2x2()
{
    std::array<Src, output_size> outs;    

    outs[0] = Src{1,0};
    outs[1] = Src{1,1};
    outs[2] = Src{2,0};
    outs[3] = Src{2,1};

    return outs;
}

constexpr bool test_3_core_2x2()
{
    constexpr std::size_t axon_count{2};
    constexpr std::size_t neuron_count{2};
    constexpr std::size_t core_count{3};
    constexpr std::size_t input_size{2};
    constexpr std::size_t output_size{4};
    constexpr MembraneLeak<weight_t> leak{1};

    using Src = SpikeSource;
    using Con = CoreConnection<axon_count>;

    auto chip = generate_test_chip<
        generate_connections_3_cores_2x2<Con, Src, core_count>(),
        generate_outputs_3_cores_2x2<Src, output_size>(),
        axon_count,
        neuron_count,
        core_count,
        input_size,
        output_size,
        leak,
        SpikingExecution<12, DeterministicSpikeGenerator>> ();

    using ChipW = ChipWeights<core_count, neuron_count, axon_count>;
    using CoreW = CoreWeights<neuron_count, axon_count>;
    using NeurW = NeuronWeights<axon_count>;
    using Ws = std::array<Weight<weight_t>, axon_count>;
    
    ChipW weights{{
        CoreW{{
            NeurW{Ws{2.0,0.0}, 2.0}, NeurW{Ws{0.0,3.0}, 3.0}
        }},
        CoreW{{
            NeurW{Ws{2.0,0.0}, 1.0}, NeurW{Ws{0.0,3.0}, 3.0}
        }},
        CoreW{{
            NeurW{Ws{2.0,0.0}, 2.0}, NeurW{Ws{0.0,3.0}, 2.0}
        }}
    }};
    chip.load_weights(weights);

    std::array<raw_input_t, 2> input{1.0, 1.0};
        
    auto buffer = chip.execute(input);

    chip.reset();
    return buffer == std::array<Weight<weight_t>, 4>{1, 0, 0, 1};
}

} // namespace nevresim::tests
} // namespace nevresim