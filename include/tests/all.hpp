#pragma once

#include "tests/single_core.hpp"
#include "tests/multi_core.hpp"

#include <iostream>

namespace nevresim
{
namespace tests
{

void report_result(auto title, auto test)
{
    std::cout << title << ": ";
    if(test()) 
        std::cout << "PASS\n";
    else 
        std::cout << "FAIL\n";
}

void run_all()
{
    report_result("single core", test_single_core);
    report_result("3 core 2x2", test_3_core_2x2);
}

}
}