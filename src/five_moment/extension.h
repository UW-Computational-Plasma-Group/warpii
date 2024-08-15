#pragma once

#include <deal.II/base/parameter_handler.h>
#include <deal.II/grid/tria.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include "../extensions/extension.h"

using namespace dealii;

namespace warpii {

namespace five_moment {
/**
 * Extension for FiveMoment applications.
 */
template <int dim>
class Extension : public virtual warpii::Extension,
                  public virtual GridExtension<dim> {
   public:
    Extension() {}

     // Begin preprocessor macro
#define PREPARE_BOUNDARY_FLUX_EVALUATORS_DECL(n_species) \
    /** Prepare the `FEFaceEvaluation` objects for each of the n_species species 
        to evaluate boundary fluxes at the given face batch. \
       In general, overloads of this method should plan to call `FEFaceEvaluation::reinit` \
       and `FEFaceEvaluation::gather_evaluate`, passing the `EvaluationFlags` that are \
       desired. */ \
    virtual void prepare_boundary_flux_evaluators( \
        const unsigned int face, \
        const unsigned int species_index, \
    const LinearAlgebra::distributed::Vector<double> &src, \
        std::array<FEFaceEvaluation<dim, -1, 0, 5, double>, n_species>& \
            fluid_evals);
     // End preprocessor macro
     // Begin preprocessor macro
#define PREPARE_BOUNDARY_FLUX_EVALUATORS_DECL_WITH_FIELDS(n_species) \
    /** Prepare the `FEFaceEvaluation` objects for each of the n_species species 
        and the E and B fields \
        to evaluate boundary fluxes at the given face batch. \
       In general, overloads of this method should plan to call `FEFaceEvaluation::reinit` \
       and `FEFaceEvaluation::gather_evaluate`, passing the `EvaluationFlags` that are \
       desired. */ \
    virtual void prepare_boundary_flux_evaluators( \
        const unsigned int face, \
        const unsigned int species_index, \
    const LinearAlgebra::distributed::Vector<double> &src, \
        std::array<FEFaceEvaluation<dim, -1, 0, 5, double>, n_species>& \
            fluid_evals, \
        FEFaceEvaluation<dim, -1, 0, 3, double>& E_field_evals, \
        FEFaceEvaluation<dim, -1, 0, 3, double>& B_field_evals);
     // End preprocessor macro
     // Begin preprocessor macro
#define BOUNDARY_FLUX_DECL(n_species) \
    /** Evaluate the outward flux at the given quadrature point on the given boundary. \
        Implementations of this method should generally use `FEFaceEvaluation::get_value` \
        and related methods to obtain solution values at the face. */ \
    virtual Tensor<1, 5, VectorizedArray<double>> boundary_flux( \
        const types::boundary_id boundary_id, const unsigned int q, \
        const unsigned int species_index, \
        const std::array<FEFaceEvaluation<dim, -1, 0, 5, double>, n_species>& \
            fluid_evals);
     // End preprocessor macro
     // Begin preprocessor macro
#define BOUNDARY_FLUX_DECL_WITH_FIELDS(n_species) \
    /** Evaluate the outward flux at the given quadrature point on the given boundary. \
        Implementations of this method should generally use `FEFaceEvaluation::get_value` \
        and related methods to obtain solution values at the face. */ \
    virtual Tensor<1, 5, VectorizedArray<double>> boundary_flux( \
        const types::boundary_id boundary_id, const unsigned int q, \
        const unsigned int species_index, \
        const std::array<FEFaceEvaluation<dim, -1, 0, 5, double>, n_species>& \
            fluid_evals, \
        const FEFaceEvaluation<dim, -1, 0, 3, double>& E_field_evals, \
        const FEFaceEvaluation<dim, -1, 0, 3, double>& B_field_evals);
     // End preprocessor macro

#define N_SPECIES_DECLS(n_species) \
    PREPARE_BOUNDARY_FLUX_EVALUATORS_DECL(n_species) \
    PREPARE_BOUNDARY_FLUX_EVALUATORS_DECL_WITH_FIELDS(n_species) \
    BOUNDARY_FLUX_DECL(n_species) \
    BOUNDARY_FLUX_DECL_WITH_FIELDS(n_species)

    N_SPECIES_DECLS(1)
    N_SPECIES_DECLS(2)
};

#define PREPARE_BOUNDARY_FLUX_EVALUATORS_IMPL(n_species) \
template <int dim> \
void Extension<dim>::prepare_boundary_flux_evaluators( \
    const unsigned int, \
        const unsigned int , \
    const LinearAlgebra::distributed::Vector<double> &, \
    std::array<FEFaceEvaluation<dim, -1, 0, 5, double>, n_species>&) {}

#define PREPARE_BOUNDARY_FLUX_EVALUATORS_IMPL_WITH_FIELDS(n_species) \
template <int dim> \
void Extension<dim>::prepare_boundary_flux_evaluators( \
    const unsigned int, \
        const unsigned int , \
    const LinearAlgebra::distributed::Vector<double> &, \
    std::array<FEFaceEvaluation<dim, -1, 0, 5, double>, n_species>&, \
    FEFaceEvaluation<dim, -1, 0, 3, double>&, \
    FEFaceEvaluation<dim, -1, 0, 3, double>&) { \
    AssertThrow(false, ExcMessage("extension method was not properly overridden.")); \
}

#define BOUNDARY_FLUX_IMPL(n_species) \
template <int dim> \
Tensor<1, 5, VectorizedArray<double>> Extension<dim>::boundary_flux( \
    const types::boundary_id, const unsigned int, \
        const unsigned int , \
    const std::array<FEFaceEvaluation<dim, -1, 0, 5, double>, n_species>&) { \
    AssertThrow(false, ExcMessage("extension method was not properly overridden.")); \
}

#define BOUNDARY_FLUX_IMPL_WITH_FIELDS(n_species) \
template <int dim> \
Tensor<1, 5, VectorizedArray<double>> Extension<dim>::boundary_flux( \
    const types::boundary_id, const unsigned int, \
        const unsigned int , \
    const std::array<FEFaceEvaluation<dim, -1, 0, 5, double>, n_species>&, \
    const FEFaceEvaluation<dim, -1, 0, 3, double>&, \
    const FEFaceEvaluation<dim, -1, 0, 3, double>&) { \
    AssertThrow(false, ExcMessage("extension method was not properly overridden.")); \
}

#define N_SPECIES_IMPLS(n_species) \
    PREPARE_BOUNDARY_FLUX_EVALUATORS_IMPL(n_species) \
    PREPARE_BOUNDARY_FLUX_EVALUATORS_IMPL_WITH_FIELDS(n_species) \
    BOUNDARY_FLUX_IMPL(n_species) \
    BOUNDARY_FLUX_IMPL_WITH_FIELDS(n_species)

N_SPECIES_IMPLS(1)
N_SPECIES_IMPLS(2)

}  // namespace five_moment
}  // namespace warpii
