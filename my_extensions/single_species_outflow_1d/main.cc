#include "src/five_moment/extension.h"
#include <deal.II/matrix_free/fe_evaluation.h>
#include "src/warpii.h"
#include "src/five_moment/euler.h"
#include "src/utilities.h"

using namespace dealii;

const unsigned int ELECTRONS = 0;

class EmissiveSheathExt : public warpii::five_moment::Extension<1> {
    void prepare_boundary_flux_evaluators(
            const unsigned int face,
        const unsigned int species_index,
        const LinearAlgebra::distributed::Vector<double> &src,
        std::array<FEFaceEvaluation<1, -1, 0, 5, double>, 1> &fluid_evals) override {
        auto& electrons = fluid_evals[ELECTRONS];
        if (species_index == ELECTRONS) {
            electrons.reinit(face);
            electrons.gather_evaluate(src, EvaluationFlags::values);
        }
    }

    Tensor<1, 5, VectorizedArray<double>> boundary_flux(
            const types::boundary_id , const unsigned int q,
        const unsigned int species_index,
        const std::array<FEFaceEvaluation<1, -1, 0, 5, double>, 1> &fluid_evals) override {

        Tensor<1, 5, VectorizedArray<double>> result;
        if (species_index != ELECTRONS) {
            std::cout << "Exiting early??" << std::endl;
            return result;
        }
        const auto& electrons = fluid_evals[ELECTRONS];

        const auto q_in = electrons.get_value(q);

        Tensor<1, 5, VectorizedArray<double>> ghost_state;
        ghost_state[0] = VectorizedArray<double>(1e-6);
        ghost_state[1] = VectorizedArray<double>(0.0);
        ghost_state[2] = VectorizedArray<double>(0.0);
        ghost_state[3] = VectorizedArray<double>(0.0);
        ghost_state[4] = VectorizedArray<double>(1e-6 * 1.5);

        result = warpii::five_moment::euler_numerical_flux<1, VectorizedArray<double>>(
                q_in, ghost_state, electrons.normal_vector(q), 5.0 / 3.0);
        SHOW(result);
        return result;
    }
};


int main(int argc, char** argv) {
        dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    auto ext = std::make_shared<EmissiveSheathExt>();
    warpii::Warpii warpii_obj = warpii::Warpii::create_from_cli(argc, argv, ext);
    warpii_obj.run();
}
