#!/bin/sh
# Build the macOS pilot against the existing HYPRE and isolated serial MFEM.
set -eu
PILOT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
PILOT_OUT="$PILOT_ROOT/results/application_default_100"
PILOT_SRC="$PILOT_ROOT/experiments/application_pilot"
"${MPICC:-mpicc}" -O2 -fPIC -dynamiclib \
  -I"$PILOT_ROOT/hypre/install/include" "$PILOT_SRC/native_default.c" \
  -L"$PILOT_ROOT/hypre/install/lib" -lHYPRE \
  -Wl,-rpath,"$PILOT_ROOT/hypre/install/lib" \
  -Wl,-install_name,"$PILOT_OUT/libpilot.dylib" -o "$PILOT_OUT/libpilot.dylib"
"${CXX:-clang++}" -O3 -std=c++11 -I"$PILOT_OUT/mfem" -I"$PILOT_SRC" \
  "$PILOT_SRC/heat.cpp" "$PILOT_OUT/mfem/libmfem.a" "$PILOT_OUT/libpilot.dylib" \
  -o "$PILOT_OUT/heat_pilot_500"
