SHELL := /bin/bash

$(WARPIISOFT)/deps: install-boost install-mpi $(WARPIISOFT)/deps/dealii

$(WARPIISOFT)/deps/dealii: install-boost install-mpi
	./install-dealii.sh

### Targets for downloading, building and installing OpenMPI

.PHONY: install-mpi
install-mpi:
	./install-mpi.sh

.PHONY: install-boost
install-boost:
	./install-boost.sh
