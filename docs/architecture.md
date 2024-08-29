# WarpII Architecture

This document describes the WarpII architecture at a high level for contributors.

## Executables and entry points

The core WarpII repository builds two binaries:
- An executable, `warpii`, which can be used to execute simulations described in
  input files (`.inp` files).
- A library, `libwarpii`, which can be linked to when building [extensions](##Extensions).

The main `warpii` executable is quite simple. It creates a `Warpii` object
using the `Warpii::create_from_cli` constructor, and proceeds to call `Warpii::run`.
This loads the input file, sets up the main Application, and proceeds to solve the
problem.
The `warpii` executable also exposes the `--print-parameters` command, which
can be used to print out a canonicalized input file.
This is used in the documentation build process.

By design, the main executable is quite thin, and the `Warpii` object is designed
to be usable from a normal C++ function:
- The command line options may be set directly on the `WarpiiOpts` object, accessible
  via `warpii_obj.opts`.
- The input string can be set directly, like `warpii_obj.input = ...`.
- The main "driver" methods, `Warpii::setup()` and `Warpii::run()`, are exposed separately.
- The application object is accessible through the `Warpii::get_app()` function.

The usability of the `Warpii` object means that _in-memory_ integration tests are
quite simple to write, following this general pattern:
- Create an input string describing the problem to be solved
- Optionally, define an Extension class and instantiate it
- Build the `Warpii` object and call its `run()` method to run the simulation
- Inspect the solution objects exposed by the application.
Such tests have several advantages:
- There is no need to go to a separate scripting language such as bash to drive the overall simulation
- No file system IO is required
- The solution is inspectable in the same process, so no extra post-processing step 
  is needed to inspect solution values.

## Parameters

WarpII makes extensive use of the `ParameterHandler` type provided by `dealii`.
By design, the `ParameterHandler` is supposed to be used in two steps.
First, the application declares the parameter tree that it expects, using a tree of 
function calls with signatures like
```
Solver::declare_parameters(ParameterHandler& prm);
```
At this point the `ParameterHandler` is empty, but the tree structure is fixed.
Next, the application calls `prm.parse_input(...)`, and the input file or string is
parsed.
Finally, the application is supposed to construct the objects it needs using a
second tree of function calls with signatures like
```
static Solver Solver::create_from_parameters(ParameterHandler& prm);
```
Thus, we make a single pass over the input string to parse it, and walk the
tree only once.

In this single-pass approach, note that the tree structure is fixed, so there is
no way to have the parameter structure depend dynamically on the contents of the
input.
However, there are several instances where we want to have a dynamic tree structure:
- Different apps define totally different sets of parameters
- Each species needs its own section, and we don't want to have extraneous sections
- Extensions can define parameter sections for their own purposes.
To provide this flexibility, we parse the input string in several passes.

1. In the first pass, we declare and parse the type of application:
   ```
   set Application = FiveMoment
   ...
   ```
   Several other "universal" parameters are defined and parsed in this pass: 
   `WorkDir` and `Subexpressions`.
2. The second pass declares and parses parameters that are specific to the application,
   and which affect the subsequent declarations.
   For example, the `FiveMoment` application declares `n_dims`, `n_species`, and `n_boundaries`.
3. In the third pass, we use the values of `n_species` and `n_boundaries` to declare
   the correct number of subsections, like
   ```
   subsection Species_2
       subsection BoundaryCondition_1
           set Type = Wall
       end
   end
   ```
4. The final pass is for the extension to declare any subsections indicated by the parameters
   parsed in the previous pass. For example,
   ```
   subsection Species_2
       subsection BoundaryCondition_1
           set Type = Extension
           < extension is asked to declare and parse parameters here >
       end
   end
   ```

## TODO: Five-moment application, solvers, everything else
