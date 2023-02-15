#pragma once

#include "simulator/spike_generation/stochastic_spike_generator.hpp"
#include "simulator/execution/spiking_execution.hpp"
#include "simulator/compute_policy/spiking_compute.hpp"
#include "simulator/compute_policy/real_valued_compute.hpp"
#include "simulator/compute_policy/fire_policy/default_fire.hpp"

#include "_tests/test_util.hpp"
#include "_tests/xor_test/xor_chip.hpp"

#include <iostream> 

namespace nevresim::tests
{

template <typename FirePolicy = DefaultFirePolicy>
bool test_xor()
{
    using weight_t = double;

    static constinit auto chip = 
        generate_xor_chip<SpikingCompute<FirePolicy>>();
    
    using spiking_1000 = 
        SpikingExecution<
            1000, 2, StochasticSpikeGenerator, weight_t, FirePolicy>;

    load_weights<weight_t>(
        chip, "include/_tests/xor_test/xor_chip_weights.txt");

    auto [correct, total] = test_on_inputs<spiking_1000>(
        chip,"include/_tests/xor_test/xor_inputs/xor_input_", 4);

    return (correct == total);
}

} // namespace nevresim::tests