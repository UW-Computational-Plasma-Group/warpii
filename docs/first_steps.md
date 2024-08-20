# First steps with WarpII {#first_steps}

## Installation

Currently the only way to use WarpII is to build it from source.

- Clone the repository:
```
$ git clone git@github.com:johnbcoughlin/warpii.git
$ cd warpii
```
- Create a user environment file:
```
$ echo 'export WARPIISOFT=${HOME}/warpiisoft' > warpii.user.env
```
- Select a CMake preset:
```
$ echo 'export WARPII_CMAKE_PRESET=macos' >> warpii.user.env
```
Here we selected the macos preset.
The available CMake build presets can be found in `CMakePresets.json`.

### Dependencies

WarpII has two direct dependencies: 
- The [deal.ii](https://dealii.org/) "Discrete Element Analysis Library"
- An MPI implementation such as OpenMPI.

On recent versions of MacOS, we have observed a compile error related to the version of
Boost that deal.II comes bundled with, so it is best to install a recent version of boost
to build against.

**Note!**: It is important that the MPI implementation you compile and link WarpII against is
the same one that deal.ii is compiled and linked against.

The recommended dependency installation steps are as follows:
1. Install `boost` and `openmpi` via your operating system's package manager:
```
# Macos
$ brew install openpmi boost
# Ubuntu
$ apt-get install openmpi-bin libopenmpi-dev libboost-all-dev
```
2. Install deal.ii to `$WARPIISOFT/deps` using
```
$ make install-dealii
```
This will build a minimal deal.ii library from source. It will find the MPI implementation
you installed automatically.
If you're on Ubuntu or a similar system, you can install deal.ii from a repository following
[these instructions](https://github.com/dealii/dealii/wiki/Getting-deal.II#linux-packages).
Note that the version installed this way will be quite large.

### Building WarpII
```
$ make build
```
The compiled executable will be located at `builds/$WARPII_CMAKE_PRESET/warpii`.

## Running a simple simulation

In this tutorial we run a simple single-species, five-moment simulation.
The equations we're solving are also called the compressible Euler equations.

WarpII uses deal.ii's [ParameterHandler](https://dealii.org/developer/doxygen/deal.II/classParameterHandler.html) class for user input.
The following example input initializes a single-fluid five-moment (Euler) simulation in one dimension.

`sine_wave.inp`:
```
set Application = FiveMoment
set n_dims = 1
set t_end = 1.0
set fields_enabled = false

# Use degree 2 finite elements, i.e. quadratic shape functions.
set fe_degree = 2

subsection geometry
    set left = 0.0
    set right = 1.0
    set nx = 50
end

subsection Species_1
    subsection InitialCondition
        # We will specify primitive variables: [rho, u_x, u_y, u_z, p].
        set VariablesType = Primitive
        set Function constants = pi=3.1415926535

        # The initial condition is specified using a parsed function.
        # The function components are separated by semicolons.
        set Function expression = 1 + 0.6 * sin(2*pi*x); 1.0; 0.0; 0.0; 1.0
    end
end
```

To run the simulation,
```
$ warpii sine_wave.inp
```
WarpII creates a directory to run the simulation and output files into:
```
$ ls FiveMoment__sine_wave/
solution_000.vtu
...
solution_010.vtu
```
