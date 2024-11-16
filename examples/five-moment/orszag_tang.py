import math

Ai = 1.0
Ae = 1/25
Zi = 1.0
Ze = 1.0

# Constant parameters from table 1
omega_pe = 1.0
c = 1.0
v_Ai = 0.0115 * c
du = 0.2 * v_Ai
eps = 0.2
beta_e = 0.1
beta_i = 0.1

de = 1.0
n0 = 1.0
di = de * math.sqrt(Ai / Ae)
L = 8*math.pi*di

# di = c / omega_pi = c / (omega_p tau) = 1 / omega_c tau
omega_c_tau = 1 / di
omega_p_tau = c * omega_c_tau

v_Ae = v_Ai / math.sqrt(Ae)
B0 = v_Ai * math.sqrt(Ai * n0)

# beta = p / (B_0^2/2), so p = 0.5 * beta * B_0^2
p_e = 0.5 * beta_e * B0**2
p_i = 0.5 * beta_i * B0**2

T_e = p_e / n0
T_i = p_i / n0

tau_0 = L / (2*math.pi*du)

kx = 2*math.pi / L
ky = 2*math.pi / L

omega_ce = omega_c_tau * Ze * B0 / Ae
omega_ci = omega_c_tau * Zi * B0 / Ai

params = {
    "omega_pe": omega_pe,
    "v_Ai/c": v_Ai / c,
    "L": L,
    "d_i": di,
    "omega_p_tau": omega_p_tau,
    "omega_c_tau": omega_c_tau,
    "B0": B0,
    "v_Ae": v_Ae,
    "p_e": p_e,
    "p_i": p_i,
    "omega_ce [omega_pe]": omega_ce / omega_pe,
    "omega_ci [omega_pe]": omega_ci / omega_pe
}

vx = f"{-du}*sin({ky}*y)"
vy = f"{du}*sin({kx}*x)"
vz = "0"
Bx = f"-0.2*{B0}*sin({ky}*y)"
By = f"0.2*{B0}*sin(2*{kx}*x)"
Bz = f"{B0}"

Ex = f"-({vy}*{Bz}) + ({vz}*{By})"
Ey = f"({vx}*{Bz}) - ({vz}*{Bx})"
Ez = f"-({vx}*{By}) + ({vy}*{Bx})"

input = f"""
set Application = FiveMoment
set n_dims = 2
set fe_degree = 2
set n_species = 2
set n_boundaries = 0
set write_output = true
set n_writeout_frames = 100

set ExplicitIntegrator = SSPRK2

set t_end = {0.1*tau_0}

subsection geometry
    set left = 0.0,0.0
    set right = {L},{L}
    set nx = 40,40
end

subsection Normalization
    set omega_p_tau = {omega_p_tau}
    set omega_c_tau = {omega_c_tau}
end

set fields_enabled = true

subsection PHMaxwellFields
    set phmaxwell_chi = 0
    set phmaxwell_gamma = 0
    subsection InitialCondition
        set components = {Ex}; {Ey}; {Ez}; \\
                         {Bx}; {By}; {Bz}; \\
                         0; 0
    end
end

subsection Species_0
    set name = electron
    set charge = {Ze}
    set mass = {Ae}
    subsection InitialCondition
        set VariablesType = Primitive
        set components = {Ae}*{n0}; {vx}; {vy}; \\
                         0.2*{B0}/({Ze}*{n0}*{omega_c_tau}) * (2*{kx}*cos(2*{kx}*x) + {ky}*cos({ky}*y)); \\
                         {p_e}
    end
end

subsection Species_1
    set name = ion
    set charge = {Ai}
    set mass = {Zi}
    subsection InitialCondition
        set VariablesType = Primitive
        set components = {Ai}*{n0}; {vx}; {vy}; 0; {p_i}
    end
end
"""

print(input)

#for k in params:
    #print(f"{k}: {params[k]}")

