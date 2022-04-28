#include "demo_menu.hpp"

#include <iostream> 
#include <ranges> 
#include <fstream>

#include "neuron.hpp"
#include "core.hpp"
#include "chip.hpp"
#include "input_loader.hpp"
#include "spike_generator.hpp"
#include "chip_executor.hpp"

#include "generated/generate_chip.hpp"

void print_prediction_summary(const auto& buffer)
{
    for(int q{}; auto x : buffer)
    {
        std::cout << "[" << q++ << ": " << x << "] ";
    }
    std::cout << "\n";
}

int argmax(const auto& buffer)
{
    return std::distance(
        buffer.begin(),
        std::max_element(buffer.begin(), buffer.end()));
}

void report_and_advance(
    int guess, int target, int idx, int& correct, int& total)
{
    total++;
    if(guess==target) correct++;

    std::cout 
        << idx << " guess: " << guess
        << "\n";

    std::cout 
        << idx << " target: " << target
        << "\n";

    std::cout << correct << "/" << total << "\n";

    std::cout << "\n";
}

void test_generated_chip()
{
    auto chip = nevresim::generate_chip();
    auto compute = chip.generate_compute();
    auto read_output_buffer = chip.generate_read_output_buffer();

    std::ifstream weights_stream("include/generated/chip_weights.txt");
    if(weights_stream.is_open())
    {
        weights_stream >> chip;
    }

    int correct{};
    int total{};
    for(int idx = 0; idx < 50; ++idx)
    {
        nevresim::InputLoader loader{};
        std::string fname = 
            std::string{"inputs/inputs"} +
            std::to_string(idx) +
            std::string{".txt"};

        std::ifstream input_stream(fname);
        if(input_stream.is_open())
        {
            input_stream >> loader;
        }
        
        auto buffer = 
            nevresim::ChipExecutor<nevresim::SpikingExecution>::execute(
                loader, chip, compute, read_output_buffer
            );

        chip.reset();

        print_prediction_summary(buffer);
        int guess = argmax(buffer);
        report_and_advance(guess, loader.target_, idx, correct, total);
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
    main_menu.add_menu_item({test_generated_chip, "test generated chip"});
    main_menu.show();

    return 0;
}