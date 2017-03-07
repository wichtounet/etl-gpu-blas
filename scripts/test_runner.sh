#!/bin/bash
set -e

function egblas_run {
    make clean
    make -j3 debug/bin/egblas_test

    ./debug/bin/egblas_test --reporter=junit --out catch_report_${1}.xml

    gcovr -x -b -r . --object-directory=debug/test > coverage_${1}_raw.xml
    cov_clean coverage_${1}_raw.xml coverage_${1}.xml
}

echo "Test 1. NVCC (default)"

egblas_run 1

echo "Merge the coverage reports"

merge-xml-coverage.py -o coverage_report.xml coverage_1.xml
cp catch_report_1.xml catch_report.xml
