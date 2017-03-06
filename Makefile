default: release

.PHONY: default release debug all clean test debug_test release_debug_test release_test
.PHONY: valgrind_test benchmark cppcheck coverage coverage_view format modernize tidy tidy_all doc
.PHONY: full_bench

include make-utils/flags-gpu.mk
include make-utils/cpp-utils.mk

# Be stricter
CXX_FLAGS += -Xcompiler -Werror -Xcompiler -Winvalid-pch -Xcompiler -Wno-uninitialized

# Add includes
CXX_FLAGS += -Ilib/include

# Support for extra flags
CXX_FLAGS += $(EXTRA_CXX_FLAGS)

# Support for CUDA
CXX_FLAGS += $(shell pkg-config --cflags cuda)
LD_FLAGS += $(shell pkg-config --libs cuda)

# Compile folders
$(eval $(call folder_compile_gpu,src))
$(eval $(call folder_compile_gpu,test/src))

# Collect files for the test executable
CPP_TEST_FILES=$(wildcard test/src/*.cpp)
CPP_SRC_FILES=$(wildcard src/*.cu)


TEST_FILES=$(CPP_TEST_FILES) ${CPP_SRC_FILES}

info:
	echo ${CPP_TEST_FILES}
	echo ${CPP_SRC_FILES}
	echo ${TEST_FILES}

# Create executables
$(eval $(call add_executable,egblas_test,$(TEST_FILES)))
$(eval $(call add_executable_set,egblas_test,egblas_test))

release: release_egblas_test
release_debug: release_debug_egblas_test
debug: debug_egblas_test

all: release release_debug debug

debug_test: debug_egblas_test
	./debug/bin/egblas_test

release_debug_test: release_debug_egblas_test
	./release_debug/bin/egblas_test

release_test: release_egblas_test
	./release/bin/egblas_test

test: all
	./debug/bin/egblas_test
	./release_debug/bin/egblas_test
	./release/bin/egblas_test

valgrind_test: debug
	valgrind --leak-check=full ./debug/bin/egblas_test

doc:
	doxygen Doxyfile

clean: base_clean
	rm -rf reports
	rm -rf latex/ html/

version:
	@echo `git rev-parse HEAD`

tag:
	@echo `git rev-list HEAD --count`-`git rev-parse HEAD`

include make-utils/cpp-utils-finalize.mk
