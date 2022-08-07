#include "demo_menu.hpp"

#include <iostream> 
#include <ranges> 
#include <fstream>

#include "neuron.hpp"
#include "core.hpp"
#include "chip.hpp"
#include "input_loader.hpp"
#include "weights_loader.hpp"
#include "spike_generator.hpp"
#include "chip_executor.hpp"

#include "tests/all.hpp"

#include "generated/generate_chip.hpp"

void test_generated_chip()
{
    static constinit auto chip = nevresim::generate_chip();
    constexpr auto compute = chip.generate_compute();
    constexpr auto read_output_buffer = chip.generate_read_output_buffer();

    nevresim::WeightsLoader<
        chip.core_count_, chip.neuron_count_, chip.axon_count_> 
        weights_loader{};
    std::ifstream weights_stream("include/generated/chip_weights.txt");
    if(weights_stream.is_open())
    {
        weights_stream >> weights_loader;
    }

    chip.load_weights(weights_loader.chip_weights_);

    int correct{};
    int total{};
    for(int idx = 0; idx < 50; ++idx)
    {
        nevresim::InputLoader input_loader{};
        std::string fname = 
            std::string{"inputs/inputs"} +
            std::to_string(idx) +
            std::string{".txt"};

        std::ifstream input_stream(fname);
        if(input_stream.is_open())
        {
            input_stream >> input_loader;
        }
        
        auto buffer = 
            nevresim::ChipExecutor<nevresim::SpikingExecution<
                1000, nevresim::SpikeGenerator
                >>::execute(
                input_loader.input_, chip, compute, read_output_buffer
            );

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
    main_menu.add_menu_item({test_generated_chip, "test generated chip"});
    main_menu.show();

    return 0;
}