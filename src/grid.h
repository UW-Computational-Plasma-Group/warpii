#pragma once

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/patterns.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <memory>
#include <fstream>
#include <iostream>
#include "grid_descriptions.h"

namespace warpii {
using namespace dealii;

class GridWrapper {
    public:
    static void declare_parameters(ParameterHandler& prm);
};

template <int dim>
class Grid {
   public:
    Grid(std::unique_ptr<GridDescription<dim>> description)
        : description(std::move(description)) {}

    static void declare_parameters(ParameterHandler& prm);

    static std::shared_ptr<Grid<dim>> create_from_parameters(
        ParameterHandler& prm);

    void reinit();

    Triangulation<dim> triangulation;

    void output_svg(std::string filename);

   private:
    std::unique_ptr<GridDescription<dim>> description;
};

template <int dim>
void Grid<dim>::declare_parameters(ParameterHandler& prm) {
    using ArrayPattern = Patterns::Tools::Convert<
                              std::array<unsigned int, dim>>;
    using PointPattern = Patterns::Tools::Convert<Point<dim>>;
    prm.enter_subsection("geometry");
    std::string grid_type = prm.get("GridType");
    if (grid_type == "HyperRectangle") {
        HyperRectangleDescription<dim>::declare_parameters(prm);
    } else if (grid_type == "ForwardFacingStep") {
        ForwardFacingStepDescription<dim>::declare_parameters(prm);
    } else {
        Assert(false, ExcMessage("No declaration for grid type"));
    }
    prm.leave_subsection();  // geometry
}

template <int dim>
std::shared_ptr<Grid<dim>> Grid<dim>::create_from_parameters(
    ParameterHandler& prm) {
    prm.enter_subsection("geometry");
    std::string grid_type = prm.get("GridType");
    std::shared_ptr<Grid<dim>> result;
    if (grid_type == "HyperRectangle")
    {
         result = std::make_shared<Grid<dim>>(
                HyperRectangleDescription<dim>::create_from_parameters(prm));
    } else if (grid_type == "ForwardFacingStep") {
        result = std::make_shared<Grid<dim>>(
                ForwardFacingStepDescription<dim>::create_from_parameters(prm));
    }
    prm.leave_subsection();  // geometry
    return result;
}

template <int dim>
void Grid<dim>::reinit() {
    description->reinit(triangulation);
}

template <int dim>
void Grid<dim>::output_svg(std::string filename) {
    std::ofstream out(filename);
    GridOut grid_out;
    grid_out.write_svg(triangulation, out);
}

}  // namespace warpii
