#include "simulation_input.h"
#include "utilities.h"

SimulationInput SimulationInput::create_from_parameters(ParameterHandler &prm, std::string raw_input) {
    unsigned int max_subexpr_replacements = prm.get_integer("MaxSubexpressionReplacements");

    std::string subexpr_map_raw = prm.get("Subexpressions");
    const auto pattern = Patterns::Map(
            Patterns::Anything(), Patterns::Anything(),
            0,  std::numeric_limits<unsigned int>::max(),
            ";;", ":=");
    std::map<std::string, std::string> subexpr_map = 
        Patterns::Tools::Convert<std::map<std::string, std::string>>::to_value(
                subexpr_map_raw, pattern);

    return SimulationInput(prm, raw_input, subexpr_map, max_subexpr_replacements);
}

void SimulationInput::reparse(bool is_final) {
    prm.parse_input_from_string(raw_input, "", !is_final);
}

void SimulationInput::return_to_top_level() {
    // Clear the ParameterHandler back to the top
    // TODO: this weirdness is necessary since in deal.ii 9.5, get_current_path()
    // is a private method.
    std::vector<std::string> check_path = {"geometry"};
    while (!prm.subsection_path_exists(check_path)) {
        prm.leave_subsection();
    }
}

std::string SimulationInput::get_with_subexpression_substitutions(const std::string &key) {
    std::string value = prm.get(key);

    std::string input(value);
    std::string modified(input);
    unsigned int i = 0;
    do {
        i += 1;
        AssertThrow(i <= max_subexpression_replacements, ExcMessage(
                    "Used up the allowed number of subexpression replacements "
                    "without reaching a fixed point. Ensure that your input file subexpressions "
                    "do not contain any trivial recursive definitions such as `func: func(...)` "
                    "which will not reach a fixed point under string replacement. "
                    "You should also beware of more complex recursive definitions involving "
                    "two or more subexpressions, such as "
                    "`func_A: func_B(...), func_B: func_A(...)`.\n"
                    "If you believe your input file is well-formed, you can increase the number of "
                    "replacements via the `MaxSubexpressionReplacements` parameter."
                    ));
        for (auto& pair : subexpression_map) {
            replace_all_occurences(pair.first, pair.second, modified);
        }
        input = modified;
    } while (modified != input);

    return input;
}
