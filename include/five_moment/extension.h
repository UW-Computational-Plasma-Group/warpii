#pragma once

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/types.h>
#include <deal.II/grid/tria.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include "five_moment/cell_evaluators.h"
#include "simulation_input.h"
#include "species.h"
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

    /**
     * This function is called after the app has finished traversing the `ParameterHandler` and
     * creating objects.
     *
     * - Sets the instance field values of `species` and `gas_gamma` to the parsed values
     *   which are passed in
     * - Declares grid extension parameters
     * - Declares BC parameters parses them with `populate_bc_from_parameters`.
     */
    void prepare_extension(SimulationInput& input,
            const std::vector<std::shared_ptr<Species<dim>>>& species,
            double gas_gamma) {
        this->species = species;
        this->gas_gamma = gas_gamma;

        input.return_to_top_level();
        GridExtension<dim>::declare_parameters(input.prm);
        for (unsigned int i = 0; i < species.size(); i++) {
            input.prm.enter_subsection("Species_" + std::to_string(i));
            auto& sp = species[i];
            for (types::boundary_id boundary_id : sp->bc_map.extension_boundaries()) {
                input.prm.enter_subsection("BoundaryCondition_" + std::to_string(boundary_id));
                declare_bc_parameters(input.prm, i, boundary_id);
                input.prm.leave_subsection();
            }
            input.prm.enter_subsection("GeneralSourceTerm");
            if (input.prm.get("Type") == "Extension") {
                declare_source_parameters(input.prm, i);
            }
            input.prm.leave_subsection();
            input.prm.leave_subsection();
        }

        input.reparse(true);

        input.return_to_top_level();
        for (unsigned int i = 0; i < species.size(); i++) {
            input.prm.enter_subsection("Species_" + std::to_string(i));
            auto& sp = species[i];
            for (types::boundary_id boundary_id : sp->bc_map.extension_boundaries()) {
                input.prm.enter_subsection("BoundaryCondition_" + std::to_string(boundary_id));
                populate_bc_from_parameters(input, i, boundary_id);
                input.prm.leave_subsection();
            }
            input.prm.enter_subsection("GeneralSourceTerm");
            if (input.prm.get("Type") == "Extension") {
                populate_source_from_parameters(input, i);
            }
            input.prm.leave_subsection();
            input.prm.leave_subsection();
        }

    }

    virtual void set_time(double t);

    virtual void declare_source_parameters(ParameterHandler& prm,
            unsigned int species_index);

    virtual void populate_source_from_parameters(SimulationInput& input,
            unsigned int species_index);

    virtual void prepare_source_term_evaluators(
            const unsigned int cell,
            FiveMomentCellEvaluators<dim>& evaluators);

    virtual Tensor<1, 5, VectorizedArray<double>> source_term(
            const unsigned int q, const unsigned int species_index,
            const FiveMomentCellEvaluators<dim>& evaluators);



    /**
     * Declare any parameters required for the boundary condition.
     *
     * @param prm: Will be scoped to Species_i / BoundaryCondition_b
     */
    virtual void declare_bc_parameters(ParameterHandler& prm, 
            unsigned int species_index, types::boundary_id boundary_id);

    /**
     * Override this method to store any desired information from the `ParameterHandler`
     * in this extension object.
     *
     * The following fields on the extension object will be populated at the time that 
     * this function is called:
     * - `species`
     * - `gas_gamma`
     *
     * @param prm: Will be scoped to Species_i / BoundaryCondition_b. This should be left
     * in the same subsection that it was when it was passed in: any subsections entered
     * must be left.
     */
    virtual void populate_bc_from_parameters(SimulationInput& input,
            unsigned int species_index, types::boundary_id boundary_id);

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
        const double time, \
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
        const double time, \
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

    const Species<dim>& get_species(unsigned int i) {
        return *species.at(i);
    }

    double gas_gamma;

   private:
    std::vector<std::shared_ptr<Species<dim>>> species;
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
        const double , \
        const unsigned int , \
    const std::array<FEFaceEvaluation<dim, -1, 0, 5, double>, n_species>&) { \
    AssertThrow(false, ExcMessage("extension method was not properly overridden.")); \
}

#define BOUNDARY_FLUX_IMPL_WITH_FIELDS(n_species) \
template <int dim> \
Tensor<1, 5, VectorizedArray<double>> Extension<dim>::boundary_flux( \
    const types::boundary_id, const unsigned int, \
        const double , \
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

template <int dim>
void Extension<dim>::set_time(double ) {}

template <int dim>
void Extension<dim>::declare_bc_parameters(ParameterHandler&, 
        unsigned int , types::boundary_id) {}

template <int dim>
void Extension<dim>::populate_bc_from_parameters(SimulationInput& ,
        unsigned int , types::boundary_id ) {}

template <int dim>
void Extension<dim>::declare_source_parameters(ParameterHandler&, 
        unsigned int) {}

template <int dim>
void Extension<dim>::populate_source_from_parameters(SimulationInput& ,
        unsigned int) {}

template <int dim>
void Extension<dim>::prepare_source_term_evaluators(
        const unsigned int ,
        FiveMomentCellEvaluators<dim>&
        ) {
    AssertThrow(false, ExcMessage("extension method was not properly overridden."));
}

template <int dim>
Tensor<1, 5, VectorizedArray<double>> Extension<dim>::source_term(
        const unsigned , const unsigned int ,
        const FiveMomentCellEvaluators<dim>&
        ) {
    AssertThrow(false, ExcMessage("extension method was not properly overridden."));
}

}  // namespace five_moment
}  // namespace warpii
