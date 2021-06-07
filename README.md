# IREE-TV project

## How to build IREE-TV
Prerequisites: [CMake](https://cmake.org/download/)(>=3.15), [IREE](https://github.com/google/iree), [Z3-solver](https://github.com/Z3Prover/z3) must be installed on your computer

Configure the `CMakeLists.txt`  
> *`-- snip --`*  
`# >>> modify directories below according to your environment <<<`  
`set(IREE_DIR "dir/to/iree") # IREE source top-level directory`  
`set(IREE_BUILD_DIR "dir/to/built/iree") # built IREE top-level directory`  
`set(Z3_DIR "dir/to/z3") # Z3 source top-level directory`  
*`-- snip --`*  

Create a `build` directory
> `mkdir build`

Move to the `build` directory
> `cd build`

Configure CMake
> `cmake ..`

Build the program
> `cmake --build .`

The `iree-tv` executable will be created inside `build` directory

### Release build
If you wish to build optimized version, follow these steps instead:  
Create and move to `release` directory
> `mkdir release`  
`cd release`

Configure CMake, this time in release mode
> `cmake -DCMAKE_BUILD_TYPE=RELEASE ..`

Build the program
> `cmake --build .`

## How to run IREE-TV
Run the built `iree-tv` executable as following:
> `iree-tv <.mlir before opt> <.mlir after opt>`
