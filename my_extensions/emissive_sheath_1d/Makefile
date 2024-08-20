FILENAME = main

DEAL_II_DEBUG = deal_II.g
DEAL_II_RELEASE = deal_II
DEAL_II_LIB = $(DEAL_II_RELEASE)

all: $(FILENAME)

$(FILENAME): $(FILENAME).o
	 /usr/bin/clang++ -Wall -Wextra -Wpedantic -Werror -O3 -DNDEBUG -arch arm64 -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX14.4.sdk -Wl,-search_paths_first -Wl,-headerpad_max_install_names $(FILENAME).o -o $(FILENAME)  /Users/jackcoughlin/src/warpii/builds/macos/src/liblibwarpii.a /Users/jackcoughlin/warpiisoft/deps/dealii/dealii-9.5.1-RelWithDebInfo/lib/libdeal_II.9.5.1.dylib -lm -ldl -framework Accelerate /opt/homebrew/Cellar/open-mpi/5.0.3_1/lib/libmpi_usempif08.dylib /opt/homebrew/Cellar/open-mpi/5.0.3_1/lib/libmpi_usempi_ignore_tkr.dylib /opt/homebrew/Cellar/open-mpi/5.0.3_1/lib/libmpi_mpifh.dylib /opt/homebrew/Cellar/open-mpi/5.0.3_1/lib/libmpi.dylib /Library/Developer/CommandLineTools/SDKs/MacOSX14.4.sdk/usr/lib/libz.tbd /opt/homebrew/lib/libboost_iostreams-mt.dylib /opt/homebrew/lib/libboost_serialization-mt.dylib /opt/homebrew/lib/libboost_system-mt.dylib /opt/homebrew/lib/libboost_thread-mt.dylib /opt/homebrew/lib/libboost_regex-mt.dylib /opt/homebrew/lib/libboost_chrono-mt.dylib /opt/homebrew/lib/libboost_atomic-mt.dylib

$(FILENAME).o: $(FILENAME).cc
	 /usr/bin/clang++ -DNDEBUG -I/Users/jackcoughlin/src/warpii -isystem /Users/jackcoughlin/warpiisoft/deps/dealii/dealii-9.5.1-RelWithDebInfo/include -isystem /Users/jackcoughlin/warpiisoft/deps/dealii/dealii-9.5.1-RelWithDebInfo/include/deal.II/bundled -isystem /opt/homebrew/Cellar/open-mpi/5.0.3_1/include -isystem /opt/homebrew/include -Wall -Wextra -Wpedantic -Werror -O3 -DNDEBUG -std=gnu++17 -arch arm64 -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX14.4.sdk -pedantic -Wall -Wextra -Wmissing-braces -Woverloaded-virtual -Wpointer-arith -Wsign-compare -Wsuggest-override -Wswitch -Wsynth -Wwrite-strings -Wno-deprecated-declarations -Wno-psabi -Wfloat-conversion -Qunused-arguments -Wno-unsupported-friend -Wno-pass-failed -Wno-unused-local-typedefs -openmp-simd -ftemplate-depth=1024 -O2 -funroll-loops -fstrict-aliasing -o $(FILENAME).o -c $(FILENAME).cc

clean:
	rm $(FILENAME) $(FILENAME).o

.PHONY: all clean
