#pragma once

#include <deal.II/lac/vector.h>
#include <deal.II/lac/la_parallel_vector.h>

using namespace dealii;

namespace warpii {
namespace maxwell {
class MaxwellSolutionVec {
   public:
    LinearAlgebra::distributed::Vector<double> mesh_sol;
    Vector<double> boundary_integrated_normal_poynting_vectors;

    void reinit(const MaxwellSolutionVec& other);

    void swap(MaxwellSolutionVec& other);

    void sadd(double s, double a, MaxwellSolutionVec V);
};

}  // namespace maxwell
}  // namespace warpii
