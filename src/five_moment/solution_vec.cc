#include "solution_vec.h"

namespace warpii {
    namespace five_moment {
        void FiveMSolutionVec::reinit(const FiveMSolutionVec& other) {
            mesh_sol.reinit(other.mesh_sol);
            for (unsigned int i = 0; i < other.boundary_integrated_fluxes.size(); i++) {
                boundary_integrated_fluxes.emplace_back();
                boundary_integrated_fluxes.at(i).reinit(other.boundary_integrated_fluxes.at(i));
            }
            boundary_integrated_normal_poynting_vectors.reinit(other.boundary_integrated_normal_poynting_vectors);
        }

        void FiveMSolutionVec::swap(FiveMSolutionVec& other) {
            mesh_sol.swap(other.mesh_sol);
            for (unsigned int i = 0; i < boundary_integrated_fluxes.size(); i++) {
                boundary_integrated_fluxes.at(i).swap(other.boundary_integrated_fluxes.at(i));
            }
            boundary_integrated_normal_poynting_vectors.swap(other.boundary_integrated_normal_poynting_vectors);
        }

        void FiveMSolutionVec::sadd(const double s, const double a, const FiveMSolutionVec& V) {
            mesh_sol.sadd(s, a, V.mesh_sol);
            for (unsigned int i = 0; i < boundary_integrated_fluxes.size(); i++) {
                if (!boundary_integrated_fluxes[i].is_empty()) {
                    boundary_integrated_fluxes[i].sadd(s, a, V.boundary_integrated_fluxes[i]);
                }
            }
            boundary_integrated_normal_poynting_vectors.sadd(s, a, 
                    V.boundary_integrated_normal_poynting_vectors);
        }

        Tensor<1, 5, double> FiveMBoundaryIntegratedFluxesVector::at_boundary(unsigned int boundary_id) {
            Tensor<1, 5, double> result;
            for (unsigned int comp = 0; comp < 5; comp++) {
                result[comp] = data[boundary_id * (5) + comp];
            }
            return result;
        }

        void FiveMBoundaryIntegratedFluxesVector::sadd(const double s, const double a, const FiveMBoundaryIntegratedFluxesVector& V) {
            data.sadd(s, a, V.data);
        }

        void FiveMBoundaryIntegratedFluxesVector::reinit(unsigned int n_boundaries, unsigned int) {
            data.reinit(n_boundaries * 5);
        }

        void FiveMBoundaryIntegratedFluxesVector::swap(FiveMBoundaryIntegratedFluxesVector& other) {
            data.swap(other.data);
        }

        void FiveMBoundaryIntegratedFluxesVector::zero() {
            data = 0.0;
        }

        void FiveMBoundaryIntegratedFluxesVector::reinit(const FiveMBoundaryIntegratedFluxesVector& other) {
            data.reinit(other.data);
        }

        bool FiveMBoundaryIntegratedFluxesVector::is_empty() {
            return data.size() == 0;
        }
    }
}
