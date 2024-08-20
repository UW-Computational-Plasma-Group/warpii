#include "src/five_moment/extension.h"
#include <deal.II/matrix_free/fe_evaluation.h>
#include "src/warpii.h"
#include "src/five_moment/euler.h"

using namespace dealii;

const unsigned int ELECTRONS = 0;
//const unsigned int IONS = 1;

class EmissiveSheathExt : public warpii::five_moment::Extension<1> {
    void prepare_boundary_flux_evaluators(
            const unsigned int face,
        const unsigned int species_index,
        const LinearAlgebra::distributed::Vector<double> &src,
        std::array<FEFaceEvaluation<1, -1, 0, 5, double>, 2> &fluid_evals,
        FEFaceEvaluation<1, -1, 0, 3, double>&,
        FEFaceEvaluation<1, -1, 0, 3, double>&) override {
        auto& electrons = fluid_evals[ELECTRONS];
        if (species_index == ELECTRONS) {
            electrons.reinit(face);
            electrons.gather_evaluate(src, EvaluationFlags::values);
        }
    }

    Tensor<1, 5, VectorizedArray<double>> boundary_flux(
            const types::boundary_id , const unsigned int q,
        const unsigned int species_index,
        const std::array<FEFaceEvaluation<1, -1, 0, 5, double>, 2> &fluid_evals,
    const FEFaceEvaluation<1, -1, 0, 3, double>&,
    const FEFaceEvaluation<1, -1, 0, 3, double>&) override {

        Tensor<1, 5, VectorizedArray<double>> result;
        if (species_index != ELECTRONS) {
            return result;
        }
        const auto& electrons = fluid_evals[ELECTRONS];

        const auto q_in = electrons.get_value(q);

        Tensor<1, 5, VectorizedArray<double>> ghost_state;
        ghost_state[0] = VectorizedArray<double>(1e-6);
        ghost_state[1] = VectorizedArray<double>(0.0);
        ghost_state[2] = VectorizedArray<double>(0.0);
        ghost_state[3] = VectorizedArray<double>(0.0);
        ghost_state[4] = VectorizedArray<double>(1e-6 * 10 * 1.5);

        const auto free_streaming_flux = warpii::five_moment::euler_numerical_flux<1, VectorizedArray<double>>(
                q_in, ghost_state, electrons.normal_vector(q), 5.0 / 3.0);

        auto j = Tensor<1, 1, VectorizedArray<double>>();
        j[0] = VectorizedArray<double>(1.0);
        const auto j_emission_out = j * electrons.normal_vector(q);
        const double Ze = -1.0;
        const double Ae = 1 / 25.0;
        const auto mass_flux_out = j_emission_out / Ze * Ae;
        Tensor<1, 5, VectorizedArray<double>> emission_flux;

        // Only specify mass flux if it represents an inflow
        emission_flux[0] = std::min(VectorizedArray<double>(0.0), mass_flux_out);

        return free_streaming_flux + emission_flux;
    }
};


int main(int argc, char** argv) {
        dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    auto ext = std::make_shared<EmissiveSheathExt>();
    warpii::Warpii warpii_obj = warpii::Warpii::create_from_cli(argc, argv, ext);
    warpii_obj.run();
}
