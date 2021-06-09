# IREE-TV project

## How to build IREE-TV
Prerequisites: [CMake](https://cmake.org/download/)(>=3.15),
[IREE](https://github.com/google/iree),
[Z3-solver](https://github.com/Z3Prover/z3)

```bash
mkdir build
cd build
# If you want to build a release version, please add -DCMAKE_BUILD_TYPE=RELEASE
cmake -DIREE_DIR=<dir/to/iree> \
      -DIREE_BUILD_DIR=<dir/to/iree-build> \
      -DZ3_DIR=<dir/to/z3> \
      ..
cmake --build .
```

## How to run IREE-TV
Run the built `iree-tv` executable as following:
```bash
iree-tv <.mlir before opt> <.mlir after opt>`
```