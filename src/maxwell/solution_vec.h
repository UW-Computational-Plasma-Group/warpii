#pragma once
#include <deal.II/lac/la_parallel_vector.h>

using namespace dealii;

namespace warpii {
namespace maxwell {
class MaxwellSolutionVec {
   public:
    LinearAlgebra::distributed::Vector<double> mesh_sol;

    void reinit(const MaxwellSolutionVec& other);

    void swap(MaxwellSolutionVec& other);
};
}  // namespace maxwell
}  // namespace warpii
