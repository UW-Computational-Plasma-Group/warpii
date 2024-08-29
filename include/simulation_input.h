#pragma once
#include <deal.II/base/parameter_handler.h>

using namespace dealii;

class SimulationInput {
    public:
    SimulationInput(
            ParameterHandler& prm,
            std::string raw_input,
            std::map<std::string, std::string> subexpression_map,
            unsigned int max_subexpression_replacements
            ):
        prm(prm), raw_input(raw_input), subexpression_map(subexpression_map),
        max_subexpression_replacements(max_subexpression_replacements)
    {}

    static SimulationInput create_from_parameters(ParameterHandler& prm, std::string raw_input);

    void reparse(bool is_final);

    /**
     * Return the `ParameterHandler` to the top level.
     */
    void return_to_top_level();

    std::string get_with_subexpression_substitutions(const std::string& key);

        ParameterHandler& prm;
        std::string raw_input;

    private:
        std::map<std::string, std::string> subexpression_map;
        unsigned int max_subexpression_replacements;
};
