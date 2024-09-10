#include "five_moment/cell_evaluators.h"
#include "utilities.h"
#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/fe_evaluation.h>


namespace warpii {
namespace five_moment {

template <int dim>
void FiveMomentCellEvaluators<dim>::ensure_species_evaluated(
    unsigned int species, unsigned int cell,
    EvaluationFlags::EvaluationFlags flags) {

    const auto state = species_eval_states.at(species);
    auto& phi = species_eval(species);

    if (state.has_value() && state->first == cell) {
        const auto state_flags = state->second;
        if ((state_flags | flags) == state_flags) {
            return;
        } else {
            phi.gather_evaluate(src, state_flags | flags);
        }
        species_eval_states[species] = std::make_optional(std::make_pair(cell, state_flags | flags));
    } else {
        phi.reinit(cell);
        phi.gather_evaluate(src, flags);
        species_eval_states[species] = std::make_optional(std::make_pair(cell, flags));
    }

}

template <int dim>
void FiveMomentCellEvaluators<dim>::ensure_fields_evaluated(
    unsigned int cell,
    EvaluationFlags::EvaluationFlags flags) {
    AssertThrow(fields_enabled, ExcMessage("Fields are not enabled for this simulation."));

    const auto state = field_eval_state;
    auto& phi = field_eval();

    if (state.has_value() && state->first == cell) {
        const auto state_flags = state->second;
        if (state_flags | flags == state_flags) {
            return;
        } else {
            phi->gather_evaluate(src, state_flags | flags);
        }
        field_eval_state = std::make_optional(std::make_pair(cell, state_flags | flags));
    } else {
        phi->reinit(cell);
        phi->gather_evaluate(src, flags);
        field_eval_state = std::make_optional(std::make_pair(cell, flags));
    }

}

template class FiveMomentCellEvaluators<1>;
template class FiveMomentCellEvaluators<2>;

}  // namespace five_moment
}  // namespace warpii
