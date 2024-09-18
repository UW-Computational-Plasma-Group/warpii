#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <deal.II/base/exceptions.h>

using namespace dealii;

class MaxwellBCBuilder {
    public:
        MaxwellBCBuilder(std::string type): type(type) {}

        MaxwellBCBuilder& with_dirichlet_components(std::string c) {
            this->dirichlet_components = c;
            return *this;
        }

        std::string to_input() {
            std::stringstream input;
            input << "        set Type = " << type << "\n";
            if (type == "Dirichlet") {
                input << "        subsection DirichletFunction\n" <<
                    "            set components = " << dirichlet_components << "\n" <<
                    "        end\n";
            }
            return input.str();
        }

    private:
        std::string type;
        std::string dirichlet_components;
};

MaxwellBCBuilder maxwell_dirichlet_bc(std::string components);

class MaxwellFieldsBuilder {
    public:
        MaxwellFieldsBuilder():
            phmaxwell_chi(0.0),
            phmaxwell_gamma(0.0) {}

        MaxwellFieldsBuilder with_bc(MaxwellBCBuilder&& bc) {
            this->bcs.push_back(std::move(bc));
            return *this;
        }

        std::string to_input() {
            std::stringstream input;
            input << "subsection PHMaxwellFields\n" <<
                "    set phmaxwell_chi = " << phmaxwell_chi << "\n" <<
                "    set phmaxwell_gamma = " << phmaxwell_gamma << "\n";
            for (unsigned int i = 0; i < bcs.size(); i++) {
                input << "    subsection BoundaryCondition_" << i << "\n" <<
                    bcs[i].to_input()
                    << "    end\n";
            }
            input << "end\n";

            return input.str();
        }

    private:
        std::vector<MaxwellBCBuilder> bcs;
        double phmaxwell_chi;
        double phmaxwell_gamma;
};

class SpeciesBCBuilder {
    public:
        SpeciesBCBuilder(std::string type): type(type) {}

    SpeciesBCBuilder& with_inflow_components(std::string inflow_components) {
        this->inflow_components = inflow_components;
        return *this;
    }

    std::string to_input() {
        std::stringstream input;
        input << "        set Type = " << type << "\n";
        if (type == "Inflow") {
            input << "        subsection InflowFunction\n" <<
                "        set components = " << inflow_components << "\n"
                "    end\n";
        }
        return input.str();
    }

    private:
    std::string type;
    std::string inflow_components;
};

SpeciesBCBuilder inflow_bc(std::string components);
SpeciesBCBuilder outflow_bc();

class SpeciesBuilder {
   public:
    SpeciesBuilder(std::string name, double mass, double charge)
        : name(name), mass(mass), charge(charge) {}

    SpeciesBuilder with_ic_rho_u_p(std::string components) {
        this->ic_components = components;
        return *this;
    }

    SpeciesBuilder with_bc(SpeciesBCBuilder&& bc) {
        bcs.push_back(std::move(bc));
        return *this;
    }

    std::string to_input() {
        AssertThrow(!ic_components.empty(), ExcMessage("You have not specified an initial condition for species `" + name + "`"));

        std::stringstream input;
        input << "    set name = " << name << "\n"
              << "    set charge = " << charge << "\n"
              << "    set mass = " << mass << "\n"
              << "    subsection InitialCondition\n"
              << "        set VariablesType = Primitive\n"
              << "        set components = " << ic_components << "\n"
              << "    end\n";

        for (unsigned int i = 0; i < bcs.size(); i++) {
            input << "    subsection BoundaryCondition_" << i << "\n" <<
                bcs[i].to_input() << "end\n";
        }
        return input.str();
    }

   private:
    std::string ic_components;
    std::vector<SpeciesBCBuilder> bcs;
    std::string name;
    double mass;
    double charge;
};

SpeciesBuilder neutrals(double mass = 1.0);
SpeciesBuilder electrons(double mass = 0.04);
SpeciesBuilder ions(double mass = 1.0, double charge = 1.0);

class FiveMoment1DBuilder {
   public:
    FiveMoment1DBuilder() : 
        fe_degree(2), 
        omega_p_tau(1.0), 
        omega_c_tau(1.0), 
    fields_enabled(false) {}

    FiveMoment1DBuilder& with_geometry(int nx, double L, bool periodic) {
        this->nx = nx;
        this->L = L;
        this->periodic = periodic;
        if (periodic) {
            n_boundaries = 0;
        } else {
            n_boundaries = 2;
        }
        return *this;
    }

    FiveMoment1DBuilder& with_fe_degree(int fe_degree) {
        this->fe_degree = fe_degree;
        return *this;
    }

    FiveMoment1DBuilder& until_time(double t_end) {
        this->t_end = t_end;
        return *this;
    }

    FiveMoment1DBuilder& with_species(SpeciesBuilder&& species) {
        this->species.push_back(std::move(species));
        return *this;
    }

    FiveMoment1DBuilder& with_omega_p_tau(double omega_p_tau) {
        this->omega_p_tau = omega_p_tau;
        return *this;
    }

    FiveMoment1DBuilder& with_omega_c_tau(double omega_c_tau) {
        this->omega_c_tau = omega_c_tau;
        return *this;
    }

    FiveMoment1DBuilder& with_fields(MaxwellFieldsBuilder&& fields) {
        this->fields = std::move(fields);
        this->fields_enabled = true;
        return *this;
    }

    std::string to_input() {
        std::stringstream input;
        input << "set Application = FiveMoment\n"
                 "set n_dims = 1\n";
        input << "set fe_degree = " << fe_degree << std::endl;
        input << "set n_species = " << species.size() << std::endl;
        input << "set n_boundaries = " << n_boundaries << std::endl;
        input << "set write_output = false\n";
        input << "set ExplicitIntegrator = SSPRK2\n";

        input << "set t_end = " << t_end << "\n";

        input << "subsection geometry\n" <<
            "    set left = 0\n" <<
            "    set right = " << L << "\n" <<
            "    set nx = " << nx << "\n";
        if (!periodic) {
            input << "    set periodic_dimensions =\n";
        }
        input << "end\n";
            
        input << "subsection Normalization\n" <<
                 "    set omega_p_tau = " << omega_p_tau << "\n" <<
                 "    set omega_c_tau = " << omega_c_tau << "\n" <<
                 "end\n";

        input << "set fields_enabled = " << (fields_enabled ? "true" : "false") << "\n";
        if (fields_enabled) {
            input << fields.to_input();
        }

        for (unsigned int i = 0; i < species.size(); i++) {
            input << "subsection Species_" << i << "\n";
            input << species[i].to_input();
            input << "end\n";
        }

        std::cout << input.str();

        return input.str();
    }

   private:
    int nx;
    double L;
    bool periodic;
    int n_boundaries;
    int fe_degree;
    double t_end;
    double omega_p_tau;
    double omega_c_tau;
    std::vector<SpeciesBuilder> species;
    bool fields_enabled;
    MaxwellFieldsBuilder fields;
};

