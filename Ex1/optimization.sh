#!/bin/bash
# =============================================================
# Run Ex1-Gasparotto-CODE.f90 with different optimization flags
# =============================================================

# Compiler
FC=gfortran
# Source code file
SRC="Ex1-Gasparotto-CODE.f90"

# Optimization flags to test
LEVELS=("O0" "O2" "O3")

# Directory to store results
OUTDIR="results"
mkdir -p "$OUTDIR"

echo "Running matrix-matrix multiplication performance tests..."
echo

# Loop over each optimization level
for LEVEL in "${LEVELS[@]}"; do
    EXE="test_${LEVEL}"
    LOG="${OUTDIR}/output_${LEVEL}.log"

    echo ">>> Compiling with -${LEVEL}..."
    $FC -${LEVEL} -fno-range-check "$SRC" -o "$EXE" || { echo "Compilation failed for -${LEVEL}"; exit 1; }

    echo ">>> Running ${EXE} ..."
    ./"${EXE}" > "$LOG"

    # Check if output data files exist before moving
    for FILE in row_col.dat col_row.dat matmul.dat; do
        if [[ -f "$FILE" ]]; then
            mv "$FILE" "${OUTDIR}/${FILE%.dat}_${LEVEL}.dat"
        fi
    done

    echo "Results saved for -${LEVEL} in ${OUTDIR}/"
    echo
done

echo "All runs completed successfully!"