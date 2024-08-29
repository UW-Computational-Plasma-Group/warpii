#pragma once
#include "app.h"
#include "../dgsem/nodal_dg_discretization.h"
#include "../wrapper.h"
#include "dg_solver.h"
#include "fields.h"

namespace warpii {
namespace maxwell {

class PHMaxwellWrapper : public ApplicationWrapper {
   public:
    void declare_parameters(ParameterHandler &prm) override;

    std::unique_ptr<AbstractApp> create_app(
        ParameterHandler &prm, std::string input,
        std::shared_ptr<warpii::Extension> ext) override;
};

template <int dim>
class PHMaxwellApp : public AbstractApp {
   public:
    PHMaxwellApp(std::shared_ptr<NodalDGDiscretization<dim>> discretization,
                 std::shared_ptr<PHMaxwellFields<dim>> fields,
                 std::shared_ptr<Grid<dim>> grid, 
                 std::shared_ptr<PHMaxwellDGSolver<dim>> dg_solver,
                 bool write_output,
                 unsigned int n_writeout_frames, double t_end)
        : discretization(discretization),
          fields(fields),
          grid(grid),
          dg_solver(dg_solver),
          write_output(write_output),
          n_writeout_frames(n_writeout_frames),
          t_end(t_end) {}

    static void declare_parameters(ParameterHandler &prm);

    static std::unique_ptr<PHMaxwellApp<dim>> create_from_parameters(
        ParameterHandler &prm);

    void setup(WarpiiOpts opts) override;

    void run(WarpiiOpts opts) override;

    void reinit(ParameterHandler &prm);

    void output_results(const unsigned int frame_number);

    const MaxwellSolutionVec& get_solution() {
        return dg_solver->get_solution();
    }

    const PHMaxwellSolutionHelper<dim>& get_solution_helper() {
        return dg_solver->get_solution_helper();
    }

   private:
    std::shared_ptr<NodalDGDiscretization<dim>> discretization;
    std::shared_ptr<PHMaxwellFields<dim>> fields;
    std::shared_ptr<Grid<dim>> grid;
    std::shared_ptr<PHMaxwellDGSolver<dim>> dg_solver;
    bool write_output;
    unsigned int n_writeout_frames;
    double t_end;
};
}  // namespace maxwell
}  // namespace warpii
