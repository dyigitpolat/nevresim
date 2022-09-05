#include "menu/demo_menu.hpp"

#include "simulator/spike_generation/stochastic_spike_generator.hpp"
#include "simulator/chip_executor.hpp"
#include "simulator/compute_policy/spiking_compute.hpp"
#include "simulator/compute_policy/real_valued_compute.hpp"

#include "_tests/all.hpp"
#include "_generated/simple_generate_chip.hpp"

#include <iostream> 

namespace nevresim
{

void test_simple_generated_chip_spiking()
{
    static constinit auto chip = 
        generate_chip<ConcreteSpikingCompute>();
    
    ChipExecutor<SpikingExecution<5000, StochasticSpikeGenerator>> executor_1;
    ChipExecutor<SpikingExecution<100000, StochasticSpikeGenerator>> executor_2;

    tests::load_weights(chip, "include/_generated/simple_chip_weights.txt");
    tests::test_on_inputs(chip,"inputs/simple_input_", 4, executor_1);
    tests::test_on_inputs(chip,"inputs/simple_input_", 4, executor_2);
}

} // namespace nevresim

int main()
{
    using namespace nevresim;
    using namespace nevresim::tests;
    
    //-----------------------------------------//
    std::cout << "- - - - - - - - - - - - - -\n";
    std::cout << "- - - - - - - - - - - - - -\n";
    std::cout << "- -   N E V R E S I M   - -\n";
    std::cout << "- - - - - - - - - - - - - -\n";
    std::cout << "- - - - - - - - - - - - - -\n";
    //-----------------------------------------//

    DemoMenu main_menu("MAIN MENU");
    
    main_menu.add_menu_item({run_all, "run all tests"});
    main_menu.add_menu_item({test_simple_generated_chip_spiking, 
        "spiking test of generated XOR chip"});
    main_menu.show();

    return 0;
}