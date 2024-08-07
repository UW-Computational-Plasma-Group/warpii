SHELL := /bin/bash

include warpii.default.env
-include warpii.user.env

check-env: required-vars-set warn-env-vars

# Check validity of key variables
required-vars-set: $(WARPIISOFT)
	@if [ ! -d "$(WARPIISOFT)" ]; then \
		echo "WARPIISOFT must be a valid directory. Current value: '$(WARPIISOFT)'"; \
		exit 1; \
	fi; \
	if ! cmake --preset=$(WARPII_CMAKE_PRESET) -N > /dev/null 2>&1 ; then \
		echo "WARPII_CMAKE_PRESET must be a valid CMake preset. Current value: '$(WARPII_CMAKE_PRESET)'"; \
		echo; \
		cmake --list-presets=configure; \
		exit 1; \
	fi


# If variables are set in the environment in addition to being set as make variables,
# it may be confusing for the user.
# This warns about such behavior.
warn-env-vars:
	@if [[ ! -z "$(shell echo $$WARPIISOFT)" && ("$(WARPIISOFT)" != "$(shell echo $$WARPIISOFT)") ]]; then \
		echo "Warning: WARPIISOFT is set as both a Make variable and an environment variable, and the values disagree."; \
		echo "Env var: $(shell echo $$WARPIISOFT)"; \
		echo "Make var: $(WARPIISOFT)"; \
		echo "The environment variable will be ignored."; \
		echo "This may be because you tried to override a default value like this:"; \
		echo; \
		echo "    WARPIISOFT=/some/dir make target"; \
		echo; \
		echo "Rather than the correct way, like this:"; \
		echo; \
		echo "    make target WARPIISOFT=/some/dir"; \
		echo; \
	fi; \
	if [[ ! -z "$(shell echo $$WARPII_CMAKE_PRESET)" && ("$(WARPII_CMAKE_PRESET)" != "$(shell echo $$WARPII_CMAKE_PRESET)") ]]; then \
		echo "Warning: WARPII_CMAKE_PRESET is set as both a Make variable and an environment variable, and the values disagree."; \
		echo "Env var: $(shell echo $$WARPII_CMAKE_PRESET)"; \
		echo "Make var: $(WARPII_CMAKE_PRESET)"; \
		echo "The environment variable will be ignored."; \
		echo "This may be because you tried to override a default value like this:"; \
		echo; \
		echo "    WARPII_CMAKE_PRESET=preset make target"; \
		echo; \
		echo "Rather than the correct way, like this:"; \
		echo; \
		echo "    make target WARPII_CMAKE_PRESET=preset"; \
		echo; \
	fi

$(WARPIISOFT):
	mkdir -p $(WARPIISOFT)

builds/$(WARPII_CMAKE_PRESET)/CMakeFiles: check-env CMakePresets.json CMakeLists.txt cmake
	cmake --preset $(WARPII_CMAKE_PRESET)

build: check-env src codes builds/$(WARPII_CMAKE_PRESET)/CMakeFiles
	source warpii.env && cmake --build --preset $(WARPII_CMAKE_PRESET) --parallel $(CMAKE_BUILD_PARALLEL_LEVEL)

test: check-env build doc
	cd builds/$(WARPII_CMAKE_PRESET) \
		&& ctest --output-on-failure -R $(WARPII_TEST_FILTER)

.PHONY: install-dealii
install-dealii: $(WARPIISOFT)
	cd script && $(MAKE) $(WARPIISOFT)/deps/dealii

doc: check-env
	$(MAKE) build \
		&& cd builds/$(WARPII_CMAKE_PRESET) \
		&& $(MAKE) documentation

.PHONY: clean check-env required-vars-set warn-env-vars
clean:
	rm -rf builds

