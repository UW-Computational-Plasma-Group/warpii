#pragma once
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/patterns.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <fstream>
#include <iomanip>
#include <memory>
#include <string>

#include "app.h"
#include "grid.h"
#include "../wrapper.h"
#include "../dgsem/nodal_dg_discretization.h"
#include "dg_solver.h"
#include "../timestepper.h"
#include "postprocessor.h"
#include "solution_vec.h"
#include "five_moment/species.h"
#include "five_moment/extension.h"
#include "dg_solution_helper.h"
#include "../common_params.h"

using namespace dealii;

namespace warpii {
namespace five_moment {

class FiveMomentWrapper : public ApplicationWrapper {
   public:
    void declare_parameters(ParameterHandler &prm) override;

    std::unique_ptr<AbstractApp> create_app(SimulationInput& input,
                                            std::shared_ptr<warpii::Extension> extension) override;
};

template <int dim>
class FiveMomentApp : public AbstractApp {
   public:
    FiveMomentApp(
        std::shared_ptr<five_moment::Extension<dim>> extension,
        std::shared_ptr<NodalDGDiscretization<dim>> discretization,
        std::vector<std::shared_ptr<Species<dim>>> species,
        std::shared_ptr<Grid<dim>> grid,
        std::unique_ptr<FiveMomentDGSolver<dim>> solver, 
        double gas_gamma,
        bool fields_enabled,
        unsigned int n_boundaries,
        bool write_output,
        double t_end,
        unsigned int n_writeout_frames
        )
        : extension(extension),
          discretization(discretization),
          species(species),
          grid(grid),
          solver(std::move(solver)),
          gas_gamma(gas_gamma),
          fields_enabled(fields_enabled),
          n_boundaries(n_boundaries),
          write_output(write_output),
          t_end(t_end),
          n_writeout_frames(n_writeout_frames)
    {}

    static void declare_parameters(ParameterHandler &prm,
            std::shared_ptr<five_moment::Extension<dim>> ext);

    static std::unique_ptr<FiveMomentApp<dim>> create_from_parameters(
            SimulationInput& input, std::shared_ptr<five_moment::Extension<dim>> ext);

    void setup(WarpiiOpts opts) override;

    void run(WarpiiOpts opts) override;

    void reinit(ParameterHandler &prm);

    NodalDGDiscretization<dim> &get_discretization();

    FiveMomentDGSolutionHelper<dim> &get_solution_helper();

    FiveMomentDGSolver<dim> &get_solver();

    FiveMSolutionVec &get_solution();

    void output_results(const double t, const unsigned int result_number);

    void append_diagnostics(const double t, const bool with_header=false);

   private:
    std::shared_ptr<five_moment::Extension<dim>> extension;
    std::shared_ptr<NodalDGDiscretization<dim>> discretization;
    std::vector<std::shared_ptr<Species<dim>>> species;
    std::shared_ptr<Grid<dim>> grid;
    std::unique_ptr<FiveMomentDGSolver<dim>> solver;
    double gas_gamma;
    bool fields_enabled;
    unsigned int n_boundaries;
    bool write_output;
    double t_end;
    unsigned int n_writeout_frames;
};

template <int dim>
void FiveMomentApp<dim>::declare_parameters(ParameterHandler &prm,
        std::shared_ptr<five_moment::Extension<dim>>) {
    unsigned int n_species = prm.get_integer("n_species");
    unsigned int n_boundaries = prm.get_integer("n_boundaries");

    PlasmaNormalization::declare_parameters(prm);

    std::vector<Species<dim>> species;
    for (unsigned int i = 0; i < n_species; i++) {
        std::stringstream subsection_name;
        subsection_name << "Species_" << i;
        prm.enter_subsection(subsection_name.str());
        Species<dim>::declare_parameters(prm, n_boundaries);
        prm.leave_subsection();
    }
    PHMaxwellFields<dim>::declare_parameters(prm, n_boundaries);

    Grid<dim>::declare_parameters(prm);

    declare_fe_degree(prm);
    prm.declare_entry("fields_enabled", "auto", Patterns::Selection("true|false|auto"),
            R"(Whether electromagnetic fields are enabled for this problem.
    - `true`: fields are enabled and will be evolved
    - `false`: fields are disabled
    - `auto`: fields are enabled if and only if `n_species >= 2`

If enabled, the solver always uses 8 components for the EM fields regardless of n_dims.
The components are
```
    [ Ex, Ey, Ez, Bx, By, Bz, phi, psi ],
```
where `phi` and `psi` are the scalar divergence error indicators for Gauss's law and the div-B law,
as used in the perfectly hyperbolic Maxwell's equation system.
            )");
    prm.declare_entry("explicit_fluid_field_coupling", "false", Patterns::Bool());
    prm.declare_entry("gas_gamma", "1.6666666666667", Patterns::Double(),
            R"(The gas gamma, AKA the ratio of specific heats, AKA `(n_dims+2)/2` for a plasma.
Defaults to 5/3, the value for simple ions with 3 degrees of freedom.)");
    declare_t_end(prm);
    declare_write_output(prm);
    declare_n_writeout_frames(prm);

    prm.declare_entry("ExplicitIntegrator", "RK1", Patterns::Selection("RK1|SSPRK2"),
            "The type of integrator to use for the explicit time marching of the flux terms.");
    prm.declare_entry("SplittingScheme", "Strang", Patterns::Selection("Strang|LieTrotter"),
            "The type of splitting scheme to use for the explicit and implicit terms.");
}

template <int dim>
std::unique_ptr<FiveMomentApp<dim>> FiveMomentApp<dim>::create_from_parameters(
        SimulationInput& input,
    std::shared_ptr<five_moment::Extension<dim>> ext) {
    ParameterHandler& prm = input.prm;

    unsigned int n_species = prm.get_integer("n_species");
    unsigned int n_boundaries = prm.get_integer("n_boundaries");

    double gas_gamma = prm.get_double("gas_gamma");

    const PlasmaNormalization plasma_norm = 
        PlasmaNormalization::create_from_parameters(input);

    std::vector<std::shared_ptr<Species<dim>>> species;
    for (unsigned int i = 0; i < n_species; i++) {
        std::stringstream subsection_name;
        subsection_name << "Species_" << i;
        prm.enter_subsection(subsection_name.str());
        species.push_back(
            Species<dim>::create_from_parameters(input, n_boundaries, gas_gamma));
        prm.leave_subsection();
    }
    auto fields = PHMaxwellFields<dim>::create_from_parameters(
            input, n_boundaries, plasma_norm);

    auto grid = Grid<dim>::create_from_parameters(prm, 
            std::static_pointer_cast<GridExtension<dim>>(ext));

    unsigned int fe_degree = prm.get_integer("fe_degree");
    std::string fields_enabled_str = prm.get("fields_enabled");
    bool fields_enabled = (fields_enabled_str == "true" || (fields_enabled_str == "auto" && n_species > 1));
    bool explicit_fluid_field_coupling = prm.get_bool("explicit_fluid_field_coupling");

    unsigned int n_field_components = fields_enabled ? 8 : 0;
    unsigned int n_components = n_species * 5 + n_field_components;
    double t_end = prm.get_double("t_end");
    bool write_output = prm.get_bool("write_output");
    unsigned int n_writeout_frames = prm.get_integer("n_writeout_frames");

    auto discretization = std::make_shared<NodalDGDiscretization<dim>>(
        grid, n_components, fe_degree);

    const auto integrator_type = prm.get("ExplicitIntegrator");
    const auto splitting_scheme = prm.get("SplittingScheme");

    auto dg_solver = std::make_unique<FiveMomentDGSolver<dim>>(
        ext, discretization, species, fields, plasma_norm, gas_gamma, 
        t_end, n_boundaries, fields_enabled, explicit_fluid_field_coupling, 
        splitting_scheme, integrator_type);


    auto app = std::make_unique<FiveMomentApp<dim>>(ext, discretization, species,
                                                    grid, std::move(dg_solver),
                                                    gas_gamma, 
                                                    fields_enabled,
                                                    n_boundaries,
                                                    write_output,
                                                    t_end,
                                                    n_writeout_frames);

    ext->prepare_extension(input, app->species, gas_gamma);

    return app;
}

template <int dim>
NodalDGDiscretization<dim> &FiveMomentApp<dim>::get_discretization() {
    return *discretization;
}

template <int dim>
FiveMomentDGSolutionHelper<dim>& FiveMomentApp<dim>::get_solution_helper() {
    return solver->get_solution_helper();
}

template <int dim>
FiveMSolutionVec &FiveMomentApp<dim>::get_solution() {
    return solver->get_solution();
}

template <int dim>
FiveMomentDGSolver<dim> &FiveMomentApp<dim>::get_solver() {
    return *solver;
}

template <int dim>
void FiveMomentApp<dim>::setup(WarpiiOpts) {
    grid->reinit();
    if (dim == 2) {
        grid->output_svg("grid.svg");
    }
    solver->reinit();
    solver->project_initial_condition();
    output_results(0.0, 0);
    append_diagnostics(0.0, true);
}

template <int dim>
void FiveMomentApp<dim>::run(WarpiiOpts) {
    double writeout_interval = t_end / n_writeout_frames;
    auto writeout = [&](double t) -> void {
        output_results(t, static_cast<unsigned int>(std::round(t / writeout_interval)));
    };
    // skip the zeroth writeout because we already did that in the setup phase
    TimestepCallback writeout_callback = TimestepCallback(writeout_interval, writeout, false);

    double diagnostic_interval = writeout_interval / 10.0;
    auto diagnostic = [&](double t) -> void {
        append_diagnostics(t);
    };
    TimestepCallback diagnostic_callback = TimestepCallback(diagnostic_interval, diagnostic, false);

    solver->solve(writeout_callback, diagnostic_callback);
}

template <int dim>
void FiveMomentApp<dim>::output_results(double time, const unsigned int result_number) {
    FiveMomentPostprocessor<dim> postprocessor(gas_gamma, species, fields_enabled);
    if (!write_output) {
        return;
    }

    DataOut<dim> data_out;

    DataOutBase::VtkFlags flags;
    flags.time = time;
    flags.write_higher_order_cells = false;
    flags.compression_level = DataOutBase::CompressionLevel::best_compression;
    data_out.set_flags(flags);

    auto& sol = get_solution();
    data_out.attach_dof_handler(discretization->get_dof_handler());
    {
        std::vector<std::string> names;
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
            interpretation;

        for (unsigned int i = 0; i < species.size(); i++) {
            auto &sp = species.at(i);
            names.emplace_back(sp->name + "_density");
            interpretation.push_back(
                DataComponentInterpretation::component_is_scalar);

            names.emplace_back(sp->name + "_x_momentum");
            names.emplace_back(sp->name + "_y_momentum");
            names.emplace_back(sp->name + "_z_momentum");
            for (unsigned int d = 0; d < 3; ++d) {
                interpretation.push_back(DataComponentInterpretation::component_is_scalar);
            }

            names.emplace_back(sp->name + "_energy");
            interpretation.push_back(
                DataComponentInterpretation::component_is_scalar);
        }
        if (fields_enabled) {
            names.emplace_back("E_x");
            names.emplace_back("E_y");
            names.emplace_back("E_z");
            for (unsigned int d = 0; d < 3; ++d) {
                interpretation.push_back(
                    DataComponentInterpretation::component_is_scalar);
            }
            names.emplace_back("B_x");
            names.emplace_back("B_y");
            names.emplace_back("B_z");
            for (unsigned int d = 0; d < 3; ++d) {
                interpretation.push_back(
                    DataComponentInterpretation::component_is_scalar);
            }
            names.emplace_back("ph_maxwell_gauss_error");
            interpretation.push_back(
                DataComponentInterpretation::component_is_scalar);
            names.emplace_back("ph_maxwell_monopole_error");
            interpretation.push_back(
                DataComponentInterpretation::component_is_scalar);
        }

        data_out.add_data_vector(discretization->get_dof_handler(),
                                 sol.mesh_sol, names, interpretation);
    }
    data_out.add_data_vector(sol.mesh_sol, postprocessor);

    Vector<double> mpi_owner(grid->triangulation.n_active_cells());
    mpi_owner = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    data_out.add_data_vector(mpi_owner, "owner");

    discretization->build_data_out_patches(data_out);

    const std::string filename =
        "solution_" + Utilities::int_to_string(result_number) + ".vtu";
    data_out.write_vtu_in_parallel(filename, MPI_COMM_WORLD);
}

template <int dim>
void FiveMomentApp<dim>::append_diagnostics(const double time, const bool with_header) {
    if (with_header) {
        std::ofstream file("diagnostics.csv");
        AssertThrow(file.is_open(), ExcMessage("Could not open diagnostics file"));
        std::string header_string = "time";
        if (fields_enabled) {
            header_string += ",electric_energy";
            header_string += ",magnetic_energy";
            
            for (unsigned int boundary_id = 0; boundary_id < n_boundaries; boundary_id++) {
                header_string += ",normal_poynting_vector_boundary_" + std::to_string(boundary_id);
            }
        }
        for (auto& sp : species) {
            header_string += "," + sp->name + "_mass";
            header_string += "," + sp->name + "_x_momentum";
            header_string += "," + sp->name + "_y_momentum";
            header_string += "," + sp->name + "_z_momentum";
            header_string += "," + sp->name + "_energy";

            for (unsigned int boundary_id = 0; boundary_id < n_boundaries; boundary_id++) {
                header_string += "," + sp->name + "_mass_flux_boundary_" + std::to_string(boundary_id);
                header_string += "," + sp->name + "_x_momentum_flux_boundary_" + std::to_string(boundary_id);
                header_string += "," + sp->name + "_y_momentum_flux_boundary_" + std::to_string(boundary_id);
                header_string += "," + sp->name + "_z_momentum_flux_boundary_" + std::to_string(boundary_id);
                header_string += "," + sp->name + "_energy_flux_boundary_" + std::to_string(boundary_id);
            }
        }

        file << header_string << std::endl;
    }

    std::ofstream file("diagnostics.csv", std::ios::app);
    AssertThrow(file.is_open(), ExcMessage("Could not open diagnostics file"));
    file << std::setprecision(16) << time;
    if (fields_enabled) {
        const auto electromagnetic_energies = get_solution_helper()
            .compute_global_electromagnetic_energy(get_solution().mesh_sol);
                
        file << "," << electromagnetic_energies[0] << "," << electromagnetic_energies[1];

        for (unsigned int boundary_id = 0; boundary_id < n_boundaries; boundary_id++) {
            file << "," << get_solution().boundary_integrated_normal_poynting_vectors(boundary_id);
        }
    }
    for (unsigned int i = 0; i < species.size(); i++) {
        const auto global_integral = get_solution_helper()
            .compute_global_integral(get_solution().mesh_sol, i);
        file << "," << global_integral[0]
            << "," << global_integral[1]
            << "," << global_integral[2]
            << "," << global_integral[3]
            << "," << global_integral[4];

            for (unsigned int boundary_id = 0; boundary_id < n_boundaries; boundary_id++) {
                const auto boundary_fluxes = get_solution().boundary_integrated_fluxes.at(i)
                    .at_boundary(boundary_id);
                file << "," << boundary_fluxes[0]
                    << "," << boundary_fluxes[1]
                    << "," << boundary_fluxes[2]
                    << "," << boundary_fluxes[3]
                    << "," << boundary_fluxes[4];
            }
    }



    file << std::endl;
}

}  // namespace five_moment
}  // namespace warpii
