#include "warpii.h"

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/patterns.h>

#include <cstdio>
#include <iostream>
#include <sstream>

#include "five_moment/five_moment.h"
#include "maxwell/maxwell_app.h"
#include "fpe.h"
#include "grid.h"
#include "opts.h"
#include "utilities.h"
#include "wrapper.h"

namespace warpii {
using namespace dealii;

char *getCmdOption(char **begin, char **end, const std::string &option) {
    char **itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) {
        return *itr;
    }
    return 0;
}

bool cmdOptionExists(char **begin, char **end, const std::string &option) {
    return std::find(begin, end, option) != end;
}

WarpiiOpts parse_opts(int argc, char **argv) {
    WarpiiOpts opts = WarpiiOpts();

    std::vector<std::string> all_args;

    // Skip the executable name by starting at 1
    for (int i = 1; i < argc; i++) {
        all_args.push_back(std::string(argv[i]));
    }

    bool set_input = false;
    for (size_t i = 0; i < all_args.size(); i++) {
        auto &arg = all_args.at(i);
        if (arg == "--help" || arg == "-h") {
            opts.help = true;
            break;
        } else if (arg == "--print-parameters") {
            opts.print_parameter_description = true;
        } else if (arg == "--enable-fpe") {
            opts.fpe = true;
        } else if (arg == "--setup-only") {
            opts.setup_only = true;
        } else {
            opts.input_is_from_file = true;
            opts.input = arg;
            set_input = true;
        }
    }

    if (!set_input && !opts.help) {
        std::cout << "Error: no input source was requested." << std::endl;
        Warpii::print_help(true);
        exit(1);
    }

    return opts;
}

Warpii Warpii::create_from_cli(int argc, char **argv) {
    WarpiiOpts opts = parse_opts(argc, argv);
    return Warpii(opts);
}

Warpii Warpii::create_from_cli(int argc, char **argv,
                               std::shared_ptr<Extension> extension) {
    WarpiiOpts opts = parse_opts(argc, argv);
    return Warpii(opts, std::move(extension));
}

// Options are specified in the docopt format: http://docopt.org/
void Warpii::print_help(bool to_err) {
    auto &stream = to_err ? std::cerr : std::cout;
    stream << R"(
WarpII: A collection of plasma codes.

Usage:
  warpii [options] <input_file>
  warpii --help | -h

Options:
  --setup-only: Defaults to false.
                If this flag is enabled, WarpII will perform only the setup() phase
                of the simulation.

  --enable-fpe: Defaults to false.
                If this flag is enabled, floating point exceptions will be enabled.
                This means that any NaNs produced during the simulation will result
                in a full stop of the simulation.
                When disabled, NaNs are silently propagated until, possibly, caught
                down the line by some dealii function.

  --print-parameters: Defaults to false.
                      If this flag is enabled, WarpII will parse the provided input file
                      and print it out.
)";
}

void Warpii::load_input_from_file() {
    std::stringstream ss;
    if (opts.input == "-") {
        ss << std::cin.rdbuf();
    } else {
        std::ifstream file(opts.input);
        if (!file.is_open()) {
            std::cerr << "Could not open requested input file <" << opts.input
                      << "> for reading." << std::endl;
            Warpii::print_help();
            exit(1);
        }
        ss << file.rdbuf();
    }
    this->input = ss.str();
}

void Warpii::setup() {
    AssertThrow(
        !setup_complete,
        ExcMessage("Cannot call Warpii::setup twice on the same object."));

    if (opts.input_is_from_file) {
        load_input_from_file();
    }

    if (opts.help) {
        Warpii::print_help();
        exit(0);
    }

    if (opts.fpe) {
        enable_floating_point_exceptions();
    }

    prm.declare_entry("Subexpressions", "", Patterns::Map(Patterns::Anything(), Patterns::Anything(), 
            0,  std::numeric_limits<unsigned int>::max(),
            ";;", ":="),
            R"(Map of subexpressions to be substituted into the input file before further processing.

Should be given in `key: value` format. For example,
```
set Subexpressions = L := 4.0;; \
                     x0 := 0.0;; \
                     T := 0.001;; \
                     gaussian(x) := exp(-((x-x0)/L)^2
```
Note that `key: value` pairs should be separated by a comma. Linebreaks can be inserted for readability
if they are escaped by a backslash `\`.

This map is used to pre-process the input file for convenience, using a simple fixed-point algorithm.
For example, if later on, the electron initial condition is specified as follows,
```
set components = gaussian(x); 0; 0; 0; T*gaussian(x)
```
then it would be processed in the following way.
After one substitution:
```
set components = exp(-((x-x0)/L)^2; 0; 0; 0; 0.001*exp(-((x-x0)/L)^2
```
After two substitutions:
```
set components = exp(-((x-0.0)/4.0)^2; 0; 0; 0; 0.001*exp(-((x-0.0)/4.0)^2
```
A third substitution leaves the string unchanged, so the algorithm stops.
            )");
    prm.declare_entry("MaxSubexpressionReplacements", "10", Patterns::Integer());

    prm.declare_entry(
        "WorkDir", "%A__%I", Patterns::Anything(),
        R"(Format string for the working directory of the simulation.

Format specifier:
    - `%A`: The name of the application, e.g. "FiveMoment"
    - `%I`: The name of the input file without file extension. If the input is taken from stdin, the specifier is replaced by the string "STDIN"
    )");
    prm.declare_entry("Application", "FiveMoment",
                      Patterns::Selection("FiveMoment|PerfectlyHyperbolicMaxwell|FPETest"));
    prm.parse_input_from_string(input, "", true);

    create_and_move_to_subdir(format_workdir(prm, opts));

    std::unique_ptr<ApplicationWrapper> app_wrapper;
    if (prm.get("Application") == "FPETest") {
        // A dummy application that immediately throws an FPE.
        // The point is to exercise the enable_floating_point_exceptions()
        // codepath in test.
        std::cout << sqrt(-1.0) << std::endl;
        exit(0);
    }
    if (prm.get("Application") == "FiveMoment") {
        app_wrapper = std::make_unique<five_moment::FiveMomentWrapper>();
    } else if (prm.get("Application") == "PerfectlyHyperbolicMaxwell") {
        app_wrapper = std::make_unique<maxwell::PHMaxwellWrapper>();
    }
    app_wrapper->declare_parameters(prm);
    SimulationInput sim_input = SimulationInput::create_from_parameters(prm, input);
    auto new_ptr = app_wrapper->create_app(sim_input, extension);
    this->app.swap(new_ptr);
    app->setup(opts);

    setup_complete = true;
}

void Warpii::run() {
    AssertThrow(!run_complete,
                ExcMessage("Cannot run the same warpii object twice."));
    if (!setup_complete) {
        setup();

        if (opts.setup_only) {
            exit(0);
        }
    }

    if (opts.help) {
        Warpii::print_help();
        exit(0);
    }
    if (opts.print_parameter_description) {
        prm.print_parameters(std::cout, ParameterHandler::OutputStyle::JSON);
        exit(0);
    }

    app->run(opts);

    run_complete = true;
}

std::string format_workdir(const ParameterHandler &prm,
                           const WarpiiOpts &opts) {
    std::string result = prm.get("WorkDir");
    size_t pos = 0;

    std::string app = prm.get("Application");
    // Loop until all occurrences of "%A" are replaced
    while ((pos = result.find("%A", pos)) != std::string::npos) {
        result.replace(pos, 2, app);
        pos += app.length();  // Move past the last replaced occurrence
    }

    pos = 0;
    std::string inp =
        (opts.input == "-") ? "STDIN" : remove_file_extension(opts.input);

    // Loop until all occurrences of "%I" are replaced
    while ((pos = result.find("%I", pos)) != std::string::npos) {
        result.replace(pos, 2, inp);
        pos += inp.length();  // Move past the last replaced occurrence
    }

    return result;
}



}  // namespace warpii
