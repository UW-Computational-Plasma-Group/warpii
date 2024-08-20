#pragma once
#include "../dgsem/nodal_dg_discretization.h"
#include "../maxwell/fields.h"
#include "solution_vec.h"
#include "species.h"
#include "../rk.h"

using namespace dealii;

namespace warpii {
namespace five_moment {

template <int dim>
class FiveMomentExplicitSourceOperator
    : public ForwardEulerOperator<FiveMSolutionVec> {
   public:
    FiveMomentExplicitSourceOperator(
        std::shared_ptr<NodalDGDiscretization<dim>> discretization,
        std::vector<std::shared_ptr<Species<dim>>> species,
        std::shared_ptr<PHMaxwellFields<dim>> fields,
        bool fields_enabled)
        : discretization(discretization),
          species(species),
          fields(fields),
          fields_enabled(fields_enabled)
    {}

    void perform_forward_euler_step(
        FiveMSolutionVec &dst, const FiveMSolutionVec &u,
        std::vector<FiveMSolutionVec> &sol_registers, const double dt,
        const double t, 
        const double b=0.0, const double a=1.0, const double c=1.0) override;

   private:
    void local_apply_inverse_mass_matrix(
        const MatrixFree<dim, double> &mf,
        LinearAlgebra::distributed::Vector<double> &dst,
        const LinearAlgebra::distributed::Vector<double> &src,
        const std::pair<unsigned int, unsigned int> &cell_range) const;

    void local_apply_cell(
        const MatrixFree<dim, double> &mf,
        LinearAlgebra::distributed::Vector<double> &dst,
        const LinearAlgebra::distributed::Vector<double> &src,
        const std::pair<unsigned int, unsigned int> &cell_range);


    std::shared_ptr<NodalDGDiscretization<dim>> discretization;
    std::vector<std::shared_ptr<Species<dim>>> species;
    std::shared_ptr<PHMaxwellFields<dim>> fields;
    bool fields_enabled;
};

}  // namespace five_moment
}  // namespace warpii
