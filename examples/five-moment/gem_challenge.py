from math import sqrt, pi

Ae = 1/25
Ai = 1
Ze = -1
Zi = 1

Lx = 8*pi
Ly = 4*pi
B0 = 0.1

# Unit speed of light, ion skin depth and inverse ion cyclotron frequency
opt = 1.0
oct = 1.0

n0 = 1.0
lambd = 0.5
n = f"{n0}*(1/5 + 1/cosh(y/{lambd})^2)"
p = f"{B0}^2 / 12 * {n}"
p_e = p
p_i = f"5*{p}"
psi0 = 0.1 * B0
dpsi_dx = f"2*pi/{Lx} * {psi0} * sin(2*pi*x / {Lx}) * cos(pi*y/{Ly})"
dpsi_dy = f"pi/{Ly} * {psi0} * cos(2*pi*x / {Lx}) * sin(pi*y/{Ly})"
deltaB_x = f"({dpsi_dy})"
deltaB_y = f"({dpsi_dx})"

Bx = f"{B0} * tanh(y/{lambd})"
By = 0
Bz = 0

Je = f"-{B0}/{lambd} * 1 / cosh(y/{lambd})^2"
u_ez = f"{Je} / ({Ze}*{n})"

input = f"""
set Application = FiveMoment
set n_dims = 2
set fe_degree = 2
set n_species = 2
set n_boundaries = 4
set write_output = true
set n_writeout_frames = 100

set ExplicitIntegrator = SSPRK2

set t_end = 100.0

subsection geometry
    set left = {-Lx/2},{-Ly/2}
    set right = {Lx/2},{Ly/2}
    set nx = 96,48
    set periodic_dimensions = x
end

subsection Normalization
    set omega_p_tau = {opt}
    set omega_c_tau = {oct}
end
    
set fields_enabled = true

subsection PHMaxwellFields
    set phmaxwell_chi = 0
    set phmaxwell_gamma = 0
    subsection InitialCondition
        set components = 0; 0; 0; \\
                         {Bx}+{deltaB_x}; {By}+{deltaB_y}; {Bz}; \\
                         0; 0
    end

    subsection BoundaryCondition_2
        set Type = PerfectConductor
    end
    subsection BoundaryCondition_3
        set Type = PerfectConductor
    end
end

subsection Species_0
    set name = electron
    set mass = {Ae}
    set charge = {Ze}
    subsection InitialCondition
        set VariablesType = Primitive
        set components = {Ae}*{n}; 0; 0; {u_ez}; {p_e}
    end

    subsection BoundaryCondition_2
        set Type = Wall
    end
    subsection BoundaryCondition_3
        set Type = Wall
    end
end

subsection Species_1
    set name = ion
    set mass = {Ai}
    set charge = {Zi}
    subsection InitialCondition
        set VariablesType = Primitive
        set components = {Ai}*{n}; 0; 0; 0; {p_i}
    end

    subsection BoundaryCondition_2
        set Type = Wall
    end
    subsection BoundaryCondition_3
        set Type = Wall
    end
end
"""

print(input)
