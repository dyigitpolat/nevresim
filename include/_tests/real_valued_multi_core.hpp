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
consteval auto generate_connections_2_cores_2x2()
{
    std::array<Con, core_count> cons; 

    cons[0].sources_[0] = Src{in,0};
    cons[0].sources_[1] = Src{in,1};
    cons[1].sources_[0] = Src{0,0};
    cons[1].sources_[1] = Src{0,1};

    return cons;
}

template <typename Src, size_t output_size>
consteval auto generate_outputs_2_cores_2x2()
{
    std::array<Src, output_size> outs;    

    outs[0] = Src{1,0};
    outs[1] = Src{1,1};

    return outs;
}

constexpr bool test_multi_real_valued_core()
{
    using weight_t = double;
    constexpr std::size_t axon_count{2};
    constexpr std::size_t neuron_count{2};
    constexpr std::size_t core_count{2};
    constexpr std::size_t input_size{2};
    constexpr std::size_t output_size{2};
    constexpr MembraneLeak<weight_t> leak{0};

    using Src = SpikeSource;
    using Con = CoreConnection<axon_count>;

    auto chip = generate_test_chip<
        weight_t,
        generate_connections_2_cores_2x2<Con, Src, core_count>(),
        generate_outputs_2_cores_2x2<Src, output_size>(),
        axon_count,
        neuron_count,
        core_count,
        input_size,
        output_size,
        leak,
        RealValuedCompute> ();

    using ChipW = ChipWeights<chip.config_, weight_t>;
    using CoreW = CoreWeights<chip.config_, weight_t>;
    using NeurW = NeuronWeights<chip.config_, weight_t>;
    using Ws = std::array<Weight<weight_t>, axon_count>;
    
    ChipW weights{{
        CoreW{{
            NeurW{Ws{0.1,0.2}}, 
            NeurW{Ws{0.5,-0.1}} 
        }},
        CoreW{{
            NeurW{Ws{0.5,-0.5}, {}, -1.0}, 
            NeurW{Ws{-1.0,1.0}, {}, 1.0} 
        }}
    }};
    chip.load_weights(weights);

    std::array<raw_input_t, 2> input{0.3, 0.2};
        
    auto buffer = chip.execute<RealValuedExecution>(input);

    chip.reset();
    return 
        std::inner_product(
            std::begin(buffer), std::end(buffer),
            std::begin(std::array<Weight<weight_t>, 2>{0.0, 1.06}), 
            true, std::logical_and<bool>{}, is_almost_equal<weight_t>);
}

} // namespace nevresim::tests
} // namespace nevresim