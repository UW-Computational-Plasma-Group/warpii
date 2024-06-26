#include "grid_descriptions.h"
#include "utilities.h"
#include "grid_generation.h"

#include <deal.II/base/patterns.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <limits>

using namespace dealii;

namespace warpii {

template <int dim>
Point<dim> default_right_point() {
    if (dim == 1) {
        return Point<dim>(1.0);
    } else if (dim == 2) {
        return Point<dim>(1.0, 1.0);
    } else {
        return Point<dim>(1.0, 1.0, 1.0);
    }
}

template <int dim>
void HyperRectangleDescription<dim>::declare_parameters(ParameterHandler& prm) {
    declare_left_right<dim>(prm);
    declare_nx<dim>(prm);
    prm.declare_entry("periodic_dimensions", "x,y,z",
                      Patterns::MultipleSelection("x|y|z"));
}

template <int dim>
std::unique_ptr<HyperRectangleDescription<dim>>
HyperRectangleDescription<dim>::create_from_parameters(ParameterHandler& prm) {
    std::array<unsigned int, dim> nx =
        Patterns::Tools::Convert<std::array<unsigned int, dim>>::to_value(
            prm.get("nx"));
    Point<dim> left =
        Patterns::Tools::Convert<Point<dim>>::to_value(prm.get("left"));
    Point<dim> right =
        Patterns::Tools::Convert<Point<dim>>::to_value(prm.get("right"));
    std::string periodic_dims = prm.get("periodic_dimensions");

    return std::make_unique<HyperRectangleDescription<dim>>(nx, left, right,
                                                            periodic_dims);
}

template <int dim>
void HyperRectangleDescription<dim>::reinit(Triangulation<dim>& triangulation) {
    std::vector<unsigned int> subdivisions =
        std::vector(nx.data(), nx.data() + dim);
    GridGenerator::subdivided_hyper_rectangle(triangulation, subdivisions, left,
                                              right, true);

    std::vector<GridTools::PeriodicFacePair<
        typename parallel::distributed::Triangulation<dim>::cell_iterator>>
        matched_pairs;
    if (periodic_dims.find("x") != std::string::npos) {
        GridTools::collect_periodic_faces(triangulation, 0, 1, 0,
                                          matched_pairs);
    }
    if (dim >= 2 && periodic_dims.find("y") != std::string::npos) {
        GridTools::collect_periodic_faces(triangulation, 2, 3, 1,
                                          matched_pairs);
    }
    if (dim >= 3 && periodic_dims.find("z") != std::string::npos) {
        GridTools::collect_periodic_faces(triangulation, 4, 5, 2,
                                          matched_pairs);
    }
    triangulation.add_periodicity(matched_pairs);
}

template class HyperRectangleDescription<1>;
template class HyperRectangleDescription<2>;

}  // namespace warpii
