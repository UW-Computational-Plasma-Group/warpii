# Getting started with WarpII {#getting_started}

The tutorials listed here introduce WarpII's capabilities in an instructional format that
is intended for users to follow along.

## 1. First steps with WarpII
[[link to tutorial](#first_steps)]

Contains instructions for installation of WarpII and its dependencies, compiling from source,
and running a simple simulation of the five-moment Euler fluid equations.
Introduces the input (`.inp`) file format and shows how to use the [ParsedFunction](https://www.dealii.org/developer/doxygen/deal.II/classFunctions_1_1ParsedFunction.html) capability
to define initial conditions on a regular rectangular grid.

## 2. Extending WarpII with C++
[[link to tutorial](#extension_tutorial)]

Introduces the C++ extension mechanism, which provides a more powerful mechanism for defining simulations
than the input file.
Contains instructions for compiling and linking a minimal extension using the `Makefile` provided by WarpII.
Shows how to construct a complex grid by populating the deal.II [Triangulation](https://www.dealii.org/developer/doxygen/deal.II/classTriangulation.html) object using library functions.
