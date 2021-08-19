# MLIR-TV project

MLIR-TV is an SMT-based translation validation framework for MLIR.
This project is inspired by [Alive2](https://github.com/aliveToolkit/alive2), an SMT-based bounded translation validation framework for LLVM IR.
However, unlike Alive2, we focus on supporting dialects that are tailored for machine learning applications only.

Currently MLIR-TV is in an experimental stage.

## How to build MLIR-TV

Prerequisites: [CMake](https://cmake.org/download/)(>=3.15),
[MLIR](https://github.com/llvm/llvm-project),
[Python3](https://www.python.org/downloads/)(>=3.9)  
Solvers (at least one of them must be used): [Z3-solver](https://github.com/Z3Prover/z3), [CVC5](https://github.com/cvc5/cvc5)

- Installation of MLIR: please follow [this instruction](https://llvm.org/docs/GettingStarted.html#getting-the-source-code-and-building-llvm) & run `cmake --build . --target install`

```bash
mkdir build
cd build
# At least one of -DZ3_DIR and -DCVC5_DIR must be provided
# -DUSE_LIBC is OFF by default. Set it to ON iff the MLIR (and CVC5) is linked against libc++
cmake -DMLIR_DIR=<dir/to/mlir-install> \
      [-DZ3_DIR=<dir/to/z3-install>] \
      [-DCVC5_DIR=<dir/to/cvc5-install>] \
      [-DUSE_LIBC=ON|OFF] \
      [-DCMAKE_BUILD_TYPE=DEBUG|RELEASE] \
      ..
cmake --build .
```

## How to run MLIR-TV
Run the built `mlir-tv` executable as following:
```bash
mlir-tv <.mlir before opt> <.mlir after opt>`
# ex: ./build/mlir-tv \
#        tests/opts/conv2d_to_img2col/nhwc_filter.src.mlir \
#        tests/opts/conv2d_to_img2col/nhwc_filter.tgt.mlir -smt-to=5000
```

To explore the semantics encoded in `mlir-tv`, you can use `mlir-interp`.
It takes a module containing functions without arguments, and prints their outputs and UBs according to the semantics encoded in it.
```bash
mlir-interp <.mlir>
# ex: ./build/mlir-interp \
#       tests/litmus/tensor-ops/extract_ub.src.mlir
```

## How to test MLIR-TV
```bash
cd build
# A detailed log is written to build/Testing/Temporary/LastTest.log
# If you want detailed output on the terminal, please add -V
ctest -R Unit
ctest -R Opts # Test IR transformation passes
ctest -R Litmus # Test litmus only
```
