#include "maxwell_app.h"

#include <deal.II/base/parameter_handler.h>

#include "../common_params.h"
#include "extensions/extension.h"
#include "dg_solver.h"
#include "maxwell/extension.h"

namespace warpii {
namespace maxwell {

void PHMaxwellWrapper::declare_parameters(ParameterHandler &prm) {
    declare_n_dims(prm);
    declare_n_boundaries(prm);
    GridWrapper::declare_parameters(prm);
}

std::unique_ptr<AbstractApp> PHMaxwellWrapper::create_app(
        SimulationInput& input,
    std::shared_ptr<warpii::Extension> extension) {
    input.reparse(false);

    switch (input.prm.get_integer("n_dims")) {
        case 1: {
            std::shared_ptr<maxwell::PHMaxwellExtension<1>> ext =
                unwrap_extension<maxwell::PHMaxwellExtension<1>>(extension);
            PHMaxwellApp<1>::declare_parameters(input.prm);
            input.reparse(true);
            return PHMaxwellApp<1>::create_from_parameters(input);
        }
        case 2: {
            std::shared_ptr<maxwell::PHMaxwellExtension<2>> ext =
                unwrap_extension<maxwell::PHMaxwellExtension<2>>(extension);
            PHMaxwellApp<2>::declare_parameters(input.prm);
            input.reparse(true);
            return PHMaxwellApp<2>::create_from_parameters(input);
        }
        default: {
            AssertThrow(false, ExcMessage("n_dims must be 1, 2, or 3"));
        }
    }
}

template <int dim>
void PHMaxwellApp<dim>::declare_parameters(ParameterHandler &prm) {
    unsigned int n_boundaries = prm.get_integer("n_boundaries");

    PlasmaNormalization::declare_parameters(prm);

    PHMaxwellFields<dim>::declare_parameters(prm, n_boundaries);

    Grid<dim>::declare_parameters(prm);

    declare_fe_degree(prm);
    declare_t_end(prm);
    declare_write_output(prm);
    declare_n_writeout_frames(prm);
}

template <int dim>
std::unique_ptr<PHMaxwellApp<dim>> PHMaxwellApp<dim>::create_from_parameters(
    SimulationInput& input) {
    ParameterHandler& prm = input.prm;
    unsigned int n_boundaries = prm.get_integer("n_boundaries");

    PlasmaNormalization plasma_norm =
        PlasmaNormalization::create_from_parameters(input);
    auto fields = PHMaxwellFields<dim>::create_from_parameters(
        input, n_boundaries, plasma_norm);
    auto grid = Grid<dim>::create_from_parameters(
        prm, std::make_shared<GridExtension<dim>>());

    unsigned int fe_degree = prm.get_integer("fe_degree");

    const double t_end = prm.get_double("t_end");
    const bool write_output = prm.get_bool("write_output");
    const unsigned int n_writeout_frames = prm.get_integer("n_writeout_frames");

    const unsigned int n_components = 8;
    auto discretization = std::make_shared<NodalDGDiscretization<dim>>(
        grid, n_components, fe_degree);

    const auto dg_solver = std::make_shared<PHMaxwellDGSolver<dim>>(
        t_end, discretization, fields, n_boundaries, plasma_norm);

    auto app = std::make_unique<PHMaxwellApp<dim>>(
        discretization, fields, grid, dg_solver, write_output, n_writeout_frames, n_boundaries, t_end);

    return app;
}

template <int dim>
void PHMaxwellApp<dim>::setup(WarpiiOpts) {
    grid->reinit();
    if (dim == 2) {
        grid->output_svg("grid.svg");
    }
    dg_solver->reinit();
    dg_solver->project_initial_condition();
    output_results(0);
}

template <int dim>
void PHMaxwellApp<dim>::run(WarpiiOpts) {
    double writeout_interval = t_end / n_writeout_frames;
    auto writeout = [&](double t) -> void {
        output_results(static_cast<unsigned int>(std::round(t / writeout_interval)));
    };
    // skip the zeroth writeout because we already did that in the setup phase
    TimestepCallback writeout_callback = TimestepCallback(writeout_interval, writeout, false);

    dg_solver->solve(writeout_callback);
}

template <int dim>
void PHMaxwellApp<dim>::output_results(const unsigned int frame_number) {
    if (!write_output) {
        return;
    }

    DataOut<dim> data_out;

    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = false;
    data_out.set_flags(flags);

    auto &sol = dg_solver->get_solution();
    data_out.attach_dof_handler(discretization->get_dof_handler());

    std::vector<std::string> names;
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        interpretation;

    names.emplace_back("Ex");
    names.emplace_back("Ey");
    names.emplace_back("Ez");
    for (unsigned int d = 0; d < 3; ++d) {
        interpretation.push_back(
            DataComponentInterpretation::component_is_scalar);
    }
    names.emplace_back("Bx");
    names.emplace_back("By");
    names.emplace_back("Bz");
    for (unsigned int d = 0; d < 3; ++d) {
        interpretation.push_back(
            DataComponentInterpretation::component_is_scalar);
    }
    names.emplace_back("ph_maxwell_gauss_error");
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    names.emplace_back("ph_maxwell_monopole_error");
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    data_out.add_data_vector(discretization->get_dof_handler(), sol.mesh_sol,
                             names, interpretation);

    Vector<double> mpi_owner(grid->triangulation.n_active_cells());
    mpi_owner = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    data_out.add_data_vector(mpi_owner, "owner");

    discretization->build_data_out_patches(data_out);

    const std::string filename =
        "solution_" + Utilities::int_to_string(frame_number, 3) + ".vtu";
    data_out.write_vtu_in_parallel(filename, MPI_COMM_WORLD);
}

template <int dim>
void PHMaxwellApp<dim>::append_diagnostics(const double time, const bool with_header) {
    if (with_header) {
        std::ofstream file("diagnostics.csv");
        AssertThrow(file.is_open(), ExcMessage("Could not open diagnostics file"));
        std::string header_string = "time";
        header_string += ",electric_energy";
        header_string += ",magnetic_energy";
        
        for (unsigned int boundary_id = 0; boundary_id < n_boundaries; boundary_id++) {
            header_string += ",normal_poynting_vector_boundary_" + std::to_string(boundary_id);
        }
    }
    std::ofstream file("diagnostics.csv", std::ios::app);
    AssertThrow(file.is_open(), ExcMessage("Could not open diagnostics file"));
    file << std::setprecision(16) << time;

    const auto electromagnetic_energies = get_solution_helper()
        .compute_global_electromagnetic_energy(get_solution().mesh_sol);
            
    file << "," << electromagnetic_energies[0] << "," << electromagnetic_energies[1];

    for (unsigned int boundary_id = 0; boundary_id < n_boundaries; boundary_id++) {
        file << "," << get_solution().boundary_integrated_normal_poynting_vectors(boundary_id);
    }
}

template class PHMaxwellApp<1>;
template class PHMaxwellApp<2>;

}  // namespace maxwell
}  // namespace warpii
