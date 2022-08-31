#pragma once

#include "_tests/single_core.hpp"
#include "_tests/multi_core.hpp"
#include "_tests/real_valued_single_core.hpp"
#include "_tests/real_valued_multi_core.hpp"

#include <iostream>

namespace nevresim
{
namespace tests
{

static_assert(test_single_core());
static_assert(test_3_core_2x2());
static_assert(test_single_real_valued_core());
static_assert(test_multi_real_valued_core());

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
    report_result("real valued single core", test_single_real_valued_core);
    report_result("real valued multi core", test_multi_real_valued_core);
}

}
}