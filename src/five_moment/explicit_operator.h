#pragma once
#include "fluid_flux_es_dgsem_operator.h"
#include "../maxwell/maxwell_flux_dg_operator.h"
#include "solution_vec.h"
#include "../rk.h"
#include "../maxwell/fields.h"
#include "explicit_source_operator.h"

using namespace dealii;

namespace warpii {

namespace five_moment {

template <int dim>
class FiveMomentExplicitOperator : public ForwardEulerOperator<FiveMSolutionVec> {
    public:
        FiveMomentExplicitOperator(
            std::shared_ptr<five_moment::Extension<dim>> extension,
            std::shared_ptr<NodalDGDiscretization<dim>> discretization,
            double gas_gamma, 
            std::vector<std::shared_ptr<Species<dim>>> species,
            std::shared_ptr<PHMaxwellFields<dim>> fields,
            PlasmaNormalization plasma_norm,
            bool fields_enabled
                ):
            fields_enabled(fields_enabled),
            fluid_flux(extension, discretization, gas_gamma, species, fields_enabled),
            explicit_sources(extension, discretization, species, fields, plasma_norm, fields_enabled),
            maxwell_flux(discretization, 5*species.size(), fields)
        {}

    TimestepResult perform_forward_euler_step(
        FiveMSolutionVec &dst,
        const FiveMSolutionVec &u,
        std::vector<FiveMSolutionVec> &sol_registers,
        const TimestepRequest dt, const double t, 
        const double b = 0.0,
        const double a = 1.0,
        const double c = 1.0) override;

    double recommend_dt(const MatrixFree<dim>& mf, const FiveMSolutionVec& soln);

    private:
    bool fields_enabled;
        FluidFluxESDGSEMOperator<dim> fluid_flux;
        FiveMomentExplicitSourceOperator<dim> explicit_sources;
        MaxwellFluxDGOperator<dim, FiveMSolutionVec> maxwell_flux;
};

}

}
