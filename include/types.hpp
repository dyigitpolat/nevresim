#pragma once

#include <cstddef>

namespace nevresim {

using weight_t = int;
using threshold_t = weight_t;
using membrane_potential_t = threshold_t;
using membrane_leak_t = membrane_potential_t;
using spike_t = std::int_fast8_t;

using core_id_t = std::size_t;
using axon_id_t = std::size_t;

} // namespace nevresim