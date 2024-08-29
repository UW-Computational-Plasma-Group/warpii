#pragma once

#include "five_moment/euler.h"
#include "species.h"
#include <deal.II/numerics/data_out.h>

#include <cmath>

namespace warpii {
namespace five_moment {
using namespace dealii;

template <int dim>
class FiveMomentPostprocessor : public DataPostprocessor<dim> {
   public:
    FiveMomentPostprocessor(double gamma,
            std::vector<std::shared_ptr<Species<dim>>> species, 
            bool fields_enabled) : gamma(gamma),
    species(species), fields_enabled(fields_enabled) {}

    virtual void evaluate_vector_field(
        const DataPostprocessorInputs::Vector<dim> &inputs,
        std::vector<Vector<double>> &computed_quantities) const override;

    virtual std::vector<std::string> get_names() const override;

    virtual std::vector<
        DataComponentInterpretation::DataComponentInterpretation>
    get_data_component_interpretation() const override;

    virtual UpdateFlags get_needed_update_flags() const override;

   private:
    double gamma;
    std::vector<std::shared_ptr<Species<dim>>> species;
    bool fields_enabled;
};

template <int dim>
void FiveMomentPostprocessor<dim>::evaluate_vector_field(
    const DataPostprocessorInputs::Vector<dim> &inputs,
    std::vector<Vector<double>> &computed_quantities) const {
    const unsigned int n_evaluation_points = inputs.solution_values.size();
    const auto points = inputs.evaluation_points;

    Assert(computed_quantities.size() == n_evaluation_points,
           ExcInternalError());
    Assert(inputs.solution_values[0].size() == 5 * species.size() + (fields_enabled ? 8 : 0), 
            ExcInternalError());
    Assert(computed_quantities[0].size() == 6 * species.size(), ExcInternalError());

    for (unsigned int p = 0; p < n_evaluation_points; ++p) {
        for (unsigned int i = 0; i < species.size(); i++) {
            Tensor<1, 5, double> solution;
            for (unsigned int comp = 0; comp < 5; ++comp) {
                solution[comp] = inputs.solution_values[p](comp + 5*i);
            }
            auto density = solution[0];
            const Tensor<1, 3> velocity = euler_velocity<3, double>(solution);
            double pressure = euler_pressure<dim, double>(solution, gamma);

            for (unsigned int d = 0; d < 3; ++d) {
                computed_quantities[p](d + 5*i) = velocity[d];
            }
            computed_quantities[p](3 + 5*i) = pressure;
            computed_quantities[p](4 + 5*i) =
                std::log(pressure) - gamma * std::log(density);
            computed_quantities[p](5 + 5*i) = std::sqrt(gamma * pressure / density);
        }
    }
}

template <int dim>
std::vector<std::string> FiveMomentPostprocessor<dim>::get_names() const {
    std::vector<std::string> names;
    for (auto& sp : species) {
        names.emplace_back(sp->name + "_x_velocity");
        names.emplace_back(sp->name + "_y_velocity");
        names.emplace_back(sp->name + "_z_velocity");
        names.emplace_back(sp->name + "_pressure");
        names.emplace_back(sp->name + "_specific_entropy");
        names.emplace_back(sp->name + "_speed_of_sound");
    }

    return names;
}

template <int dim>
UpdateFlags FiveMomentPostprocessor<dim>::get_needed_update_flags() const {
    return update_values | dealii::update_quadrature_points;
}

template <int dim>
std::vector<DataComponentInterpretation::DataComponentInterpretation>
FiveMomentPostprocessor<dim>::get_data_component_interpretation() const {
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        interpretation;
    for (unsigned int i = 0; i < species.size(); i++) {
        // velocity
        for (unsigned int d = 0; d < 3; ++d)
            interpretation.push_back(
                DataComponentInterpretation::component_is_scalar);

        // pressure
        interpretation.push_back(DataComponentInterpretation::component_is_scalar);

        // entropy
        interpretation.push_back(DataComponentInterpretation::component_is_scalar);

        // speed of sound
        interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    }

    return interpretation;
}
}  // namespace five_moment
}  // namespace warpii
