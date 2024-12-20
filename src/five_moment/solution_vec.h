#pragma once

#include <deal.II/lac/vector.h>
#include <deal.II/lac/la_parallel_vector.h>

using namespace dealii;

namespace warpii {
    namespace five_moment {
        class FiveMBoundaryIntegratedFluxesVector {
            public:
                void swap(FiveMBoundaryIntegratedFluxesVector& other);

                void zero();

                template <int dim>
                void add(unsigned int boundary_id, Tensor<1, 5, double> flux);

                void sadd(const double s, const double a, const FiveMBoundaryIntegratedFluxesVector& V);

                void reinit(unsigned int n_boundaries, unsigned int n_dims);

                void reinit(const FiveMBoundaryIntegratedFluxesVector& other);

                Tensor<1, 5, double> at_boundary(unsigned int boundary_id);

                bool is_empty();

                Vector<double> data;
        };

        template <int dim>
        void FiveMBoundaryIntegratedFluxesVector::add(unsigned int boundary_id, Tensor<1, 5, double> flux) {
            for (unsigned int comp = 0; comp < 5; comp++) {
                data[boundary_id * (5) + comp] += flux[comp];
            }
        }


        class FiveMSolutionVec {
            public:
                LinearAlgebra::distributed::Vector<double> mesh_sol;
                std::vector<FiveMBoundaryIntegratedFluxesVector> boundary_integrated_fluxes;
                Vector<double> boundary_integrated_normal_poynting_vectors;

            void reinit(const FiveMSolutionVec& other);

            void swap(FiveMSolutionVec& other);

            void sadd(const double s, const double a, const FiveMSolutionVec& V);
        };
    }
}
