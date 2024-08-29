#pragma once
#include <deal.II/base/parameter_handler.h>

using namespace dealii;

class SimulationInput {
    public:
        SimulationInput(
            ParameterHandler& prm,
            std::string raw_input):
            prm(prm), raw_input(raw_input), max_subexpression_replacements(1) {}

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

    /**
     * Re-parse the `ParameterHandler`, which may have newly declared sections and parameters,
     * from the stored raw input string.
     *
     * @param is_final: If true, we throw an exception when the parsing encounters
     * an undeclared parameter or section.
     */
    void reparse(bool is_final);

    /**
     * Return the `ParameterHandler` to the top level.
     */
    void return_to_top_level();

    /**
     * Calls `prm.get(key)`, and returns the result processed according to the subexpression
     * substitutions defined in `subexpression_map`.
     */
    std::string get_with_subexpression_substitutions(const std::string& key);

        ParameterHandler& prm;
        std::string raw_input;

    private:
        std::map<std::string, std::string> subexpression_map;
        unsigned int max_subexpression_replacements;
};
