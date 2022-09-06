#pragma once

#include "simulator/chip/neuron.hpp"
#include "simulator/chip/core.hpp"
#include "simulator/chip/chip.hpp"
#include "simulator/compute_policy/real_valued_compute.hpp"
#include "simulator/execution/real_valued_execution.hpp"

#include "test_util.hpp"

#include <sstream>

namespace nevresim {
namespace tests {

template <typename Con, typename Src, size_t core_count>
consteval auto generate_connections_single_core_3x3()
{
    std::array<Con, core_count> cons; 

    cons[0].sources_[0] = Src{in,0};
    cons[0].sources_[1] = Src{in,1};
    cons[0].sources_[2] = Src{in,2};

    return cons;
}

template <typename Src, size_t output_size>
consteval auto generate_outputs_single_core_3x3()
{
    std::array<Src, output_size> outs;    

    outs[0] = Src{0,0};
    outs[1] = Src{0,1};
    outs[2] = Src{0,2};

    return outs;
}

constexpr bool test_single_real_valued_core()
{
    constexpr std::size_t axon_count{3};
    constexpr std::size_t neuron_count{3};
    constexpr std::size_t core_count{1};
    constexpr std::size_t input_size{3};
    constexpr std::size_t output_size{3};
    constexpr MembraneLeak<weight_t> leak{0};

    using Src = SpikeSource;
    using Con = CoreConnection<axon_count>;

    auto chip = generate_test_chip<
        generate_connections_single_core_3x3<Con, Src, core_count>(),
        generate_outputs_single_core_3x3<Src, output_size>(),
        axon_count,
        neuron_count,
        core_count,
        input_size,
        output_size,
        leak,
        RealValuedCompute> ();

    using ChipW = ChipWeights<chip.config_>;
    using CoreW = CoreWeights<chip.config_>;
    using NeurW = NeuronWeights<chip.config_>;
    using Ws = std::array<Weight<weight_t>, axon_count>;
    
    ChipW weights{{
        CoreW{{
            NeurW{Ws{3.0,-2.0, 0.1}, {}, 4.2}, 
            NeurW{Ws{7.0,-5.0, 0.0}},
            NeurW{Ws{0.0, 1.0, 0.0}},
        }}
    }};
    chip.load_weights(weights);

    std::array<raw_input_t, 3> input{1.0, 0.5, 0.3};
        
    auto buffer = chip.execute<RealExecution>(input);

    chip.reset();
    return 
        std::inner_product(
            std::begin(buffer), std::end(buffer),
            std::begin(std::array<Weight<weight_t>, 3>{6.23, 4.5, 0.5}), 
            true, std::logical_and<bool>{}, is_almost_equal<double>);
}

} // namespace nevresim::tests
} // namespace nevresim