default: release_debug

.PHONY: default release debug all clean test debug_test release_debug_test release_test
.PHONY: valgrind_test tag version

include make-utils/flags-gpu.mk
include make-utils/cpp-utils.mk

# Use compute capabilities 7.0
CXX_FLAGS += -m64 -arch=sm_70

# NVCC optimizations
RELEASE_FLAGS += --use_fast_math
RELEASE_DEBUG_FLAGS += --use_fast_math

# Relocatable shared library
CXX_FLAGS += -Xcompiler -fPIC

# Fix for CUDA
CXX_FLAGS += -D_FORCE_INLINES

# Be stricter
CXX_FLAGS += -Xcompiler -Werror -Xcompiler -Winvalid-pch -Xcompiler -Wno-uninitialized -Xcompiler -Wno-unused-parameter -Xcompiler -Wno-strict-aliasing

# Add includes
CXX_FLAGS += -Ilib/include

# Support for extra flags
CXX_FLAGS += $(EXTRA_CXX_FLAGS)

# Support for CUDA
CXX_FLAGS += $(shell pkg-config --cflags cuda)
LD_FLAGS += $(shell pkg-config --libs cuda)

# Compile folders
$(eval $(call folder_compile_gpu,src))
$(eval $(call folder_compile_gpu,test/src, -ICatch/include -Itest/include))
$(eval $(call folder_compile_gpu,bench/src))

# Collect files for the test executable
CPP_TEST_FILES=$(wildcard test/src/*.cpp)
CPP_SRC_FILES=$(wildcard src/*.cu)

TEST_FILES=$(CPP_TEST_FILES) ${CPP_SRC_FILES}

# Collect files for the bench executable
BENCH_FILES=$(wildcard bench/src/*.cpp) ${CPP_SRC_FILES}

# Create test executables
$(eval $(call add_executable,egblas_test,$(TEST_FILES)))
$(eval $(call add_executable_set,egblas_test,egblas_test))

# Create bench executables
$(eval $(call add_executable,egblas_bench,$(BENCH_FILES), -lcublas -lcudnn))
$(eval $(call add_executable_set,egblas_bench,egblas_bench))

# Create the shared library
$(eval $(call add_shared_library,libegblas,$(CPP_SRC_FILES)))

release: release/lib/libegblas.so
release_debug: release_debug/lib/libegblas.so
debug: debug/lib/libegblas.so

all: release release_debug debug

debug_test: debug_egblas_test
	./debug/bin/egblas_test

release_debug_test: release_debug_egblas_test
	./release_debug/bin/egblas_test

release_test: release_egblas_test
	./release/bin/egblas_test

test: release/bin/egblas_test release_debug/bin/egblas_test debug/bin/egblas_test
	./debug/bin/egblas_test
	./release_debug/bin/egblas_test
	./release/bin/egblas_test

debug_bench: debug_egblas_bench
	./debug/bin/egblas_bench

release_debug_bench: release_debug_egblas_bench
	./release_debug/bin/egblas_bench

release_bench: release_egblas_bench
	./release/bin/egblas_bench

debug_install: debug/lib/libegblas.so
	cp -r include/* $(DESTDIR)/usr/include/
	install -m 0644 debug/lib/libegblas.so $(DESTDIR)/usr/lib/
	install -m 0644 -D egblas.pc $(DESTDIR)/usr/share/pkgconfig/

release_debug_install: release_debug/lib/libegblas.so
	cp -r include/* $(DESTDIR)/usr/include/
	install -m 0644 release_debug/lib/libegblas.so $(DESTDIR)/usr/lib/
	install -m 0644 -D egblas.pc $(DESTDIR)/usr/share/pkgconfig/

release_install: release/lib/libegblas.so
	cp -r include/* $(DESTDIR)/usr/include/
	install -m 0644 release/lib/libegblas.so $(DESTDIR)/usr/lib/
	install -m 0644 -D egblas.pc $(DESTDIR)/usr/share/pkgconfig/

install: release_debug_install

valgrind_test: debug
	valgrind --leak-check=full ./debug/bin/egblas_test

clean: base_clean
	rm -rf reports
	rm -rf latex/ html/

version:
	@echo `git rev-parse HEAD`

tag:
	@echo `git rev-list HEAD --count`-`git rev-parse HEAD`

format:
	git ls-files "*.hpp" "*.cu" "*.cpp" | xargs clang-format -i -style=file

include make-utils/cpp-utils-finalize.mk
