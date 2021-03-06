
#pragma once

#include <PscConfig.h>

#include <limits>
#include <vector>

namespace kg
{
namespace io
{

enum class Mode
{
  Write,
  Read,
  Blocking,
  NonBlocking,
};

enum class StepMode
{
  Append,
  Read,
};

using size_t = std::size_t;

using Dims = std::vector<size_t>;

// FIXME, adios2 specific (and adios2 bug, which has uint64_t)
constexpr size_t LocalValueDim = std::numeric_limits<size_t>::max() - 2;

struct Extents
{
  Dims start;
  Dims count;
};

} // namespace io
} // namespace kg

#include "io/Descr.h"
#include "io/Engine.h"
#ifdef PSC_HAVE_ADIOS2
#include "io/IOAdios2.h"
#endif
