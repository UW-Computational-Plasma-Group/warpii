#pragma once
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/evaluation_flags.h>

using namespace dealii;

namespace warpii {
namespace five_moment {

template <int dim>
std::vector<FEEvaluation<dim, -1, 0, 5, double>> construct_species_evals(
    const MatrixFree<dim, double> &mf,
    unsigned int n_species) {
    std::vector<FEEvaluation<dim, -1, 0, 5, double>> result;
    for (unsigned int i = 0; i < n_species; i++) {
        result.emplace_back(mf, 0, 1, 5*i);
    }
    return result;
}

template <int dim>
std::optional<FEEvaluation<dim, -1, 0, 8, double>> construct_field_eval(
    const MatrixFree<dim, double> &mf,
    bool fields_enabled,
    unsigned int n_species) {
    return fields_enabled
        ? std::make_optional<FEEvaluation<dim, -1, 0, 8, double>>(mf, 0, 1, 5*n_species)
        : std::nullopt;
}

template <int dim>
class FiveMomentCellEvaluators {
   public:
    FiveMomentCellEvaluators(
    const MatrixFree<dim, double> &mf,
    const LinearAlgebra::distributed::Vector<double> &src,
            unsigned int n_species, bool fields_enabled) : 
        src(src),
        n_species(n_species),
        fields_enabled(fields_enabled),
        _field_eval(construct_field_eval(mf, fields_enabled, n_species)),
        _species_evals(construct_species_evals(mf, n_species)),
        species_eval_states(n_species, std::nullopt)
    {}

    void ensure_species_evaluated(unsigned int species, unsigned int cell, 
            EvaluationFlags::EvaluationFlags flags);

    void ensure_fields_evaluated(unsigned int cell, 
            EvaluationFlags::EvaluationFlags flags);

    FEEvaluation<dim, -1, 0, 8, double>& field_eval() {
        AssertThrow(fields_enabled, ExcMessage("Fields are not enabled."));
        return *_field_eval;
    }

    FEEvaluation<dim, -1, 0, 5, double>& species_eval(unsigned int i) {
        return _species_evals.at(i);
    }

    const FEEvaluation<dim, -1, 0, 8, double>& field_eval() const {
        AssertThrow(fields_enabled, ExcMessage("Fields are not enabled."));
        return *_field_eval;
    }

    const FEEvaluation<dim, -1, 0, 5, double>& species_eval(unsigned int i) const {
        return _species_evals.at(i);
    }

   private:
    const LinearAlgebra::distributed::Vector<double> &src;
    unsigned int n_species;
    bool fields_enabled;
    std::optional<FEEvaluation<dim, -1, 0, 8, double>> _field_eval;
    std::vector<FEEvaluation<dim, -1, 0, 5, double>> _species_evals;

    std::vector<std::optional<std::pair<unsigned int, EvaluationFlags::EvaluationFlags>>>
        species_eval_states;
    std::optional<std::pair<unsigned int, EvaluationFlags::EvaluationFlags>>
        field_eval_state;
};
}  // namespace five_moment
}  // namespace warpii
