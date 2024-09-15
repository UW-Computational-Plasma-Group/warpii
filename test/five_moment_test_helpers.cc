#include "five_moment_test_helpers.h"

SpeciesBuilder electrons(double mass) {
    return SpeciesBuilder("electron", mass, -1.0);
}

SpeciesBuilder ions(double mass, double charge) {
    return SpeciesBuilder("ion", mass, charge);
}

SpeciesBCBuilder inflow_bc(std::string c) {
    return std::move(SpeciesBCBuilder("Inflow")
        .with_inflow_components(c));
}

MaxwellBCBuilder maxwell_dirichlet_bc(std::string c) {
    return MaxwellBCBuilder("Dirichlet").with_dirichlet_components(c);
}
