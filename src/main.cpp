#include "menu/demo_menu.hpp"

#include "simulator/chip/neuron.hpp"
#include "simulator/chip/core.hpp"
#include "simulator/chip/chip.hpp"
#include "loaders/input_loader.hpp"
#include "loaders/weights_loader.hpp"
#include "simulator/spike_generation/spike_generator.hpp"
#include "simulator/chip_executor.hpp"
#include "simulator/compute_policy/spiking_compute.hpp"
#include "simulator/compute_policy/real_valued_compute.hpp"

#include "_tests/all.hpp"
#include "_generated/simple_generate_chip.hpp"

#include <iostream> 
#include <ranges> 
#include <fstream>

void test_simple_generated_chip_spiking()
{
    static constinit auto chip = 
        nevresim::generate_chip<
            nevresim::SpikingExecution<
                5000, nevresim::SpikeGenerator>>();

    nevresim::WeightsLoader<
        chip.config_> 
        weights_loader{};
    std::ifstream weights_stream("include/_generated/simple_chip_weights.txt");
    if(weights_stream.is_open())
    {
        weights_stream >> weights_loader;
    }

    chip.load_weights(weights_loader.chip_weights_);

    int correct{};
    int total{};
    for(int idx = 0; idx < 4; ++idx)
    {
        nevresim::InputLoader input_loader{};
        std::string fname = 
            std::string{"inputs/simple_input_"} +
            std::to_string(idx) +
            std::string{".txt"};

        std::ifstream input_stream(fname);
        if(input_stream.is_open())
        {
            input_stream >> input_loader;
        }
        
        auto buffer = chip.execute(input_loader.input_);

        chip.reset();

        nevresim::tests::print_prediction_summary(buffer);
        int guess = nevresim::tests::argmax(buffer);
        nevresim::tests::report_and_advance(
            guess, input_loader.target_, idx, correct, total);
    }
}

int main()
{
    //-----------------------------------------//
    std::cout << "- - - - - - - - - - - - - -\n";
    std::cout << "- - - - - - - - - - - - - -\n";
    std::cout << "- -   N E V R E S I M   - -\n";
    std::cout << "- - - - - - - - - - - - - -\n";
    std::cout << "- - - - - - - - - - - - - -\n";
    //-----------------------------------------//

    DemoMenu main_menu("MAIN MENU");
    
    main_menu.add_menu_item({nevresim::tests::run_all, "run all tests"});
    main_menu.add_menu_item({test_simple_generated_chip_spiking, 
        "spiking test of generated XOR chip"});
    main_menu.show();

    return 0;
}