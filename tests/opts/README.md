# Before you add new test cases...

## Directory name convention
### `<transformation-pass-name>`
ex) `fold-tensor-extract-op`
Just strip the <case_name> and extensions away, and you get the directory name.

## Naming convention
### `<case_name>[-bad].(src|tgt).mlir`
ex) `nhwc_filter.src.mlir`, `const_tensor.tgt.mlir`, `i32-bad.src.mlir`.  
Also, each test case should form a pair of `.src.mlir` and `.tgt.mlir`.  
Use suffix `-bad` for `// VERIFY-INCORRECT` test cases

## Test keywords
`// VERIFY` : check if the transformation is correct  
`// VERIFY-INCORRECT` : check if the transformation is indeed wrong
`// UNSUPPORTED` : ignore test case that includes **yet** unimplemented dialects
`// EXPECT "<message>"` : check if the stdout/stderr includes the provided message

## And some more
Each `.src.mlir` file must include the command used to create the pair `.tgt.mlir` file. For now they are written in the first-line comment. The `iree-opt` flag sometimes isn't identical to the pass name, so reproducing the src-tgt pair becomes troublesome. Including the flag(command) will be very much appreciated for this reason ;)  
When implementing a negative test case by modifying valid `tgt.mlir`, the diff must be annotated with comments.

## Reference
https://github.com/aqjune/iree-tv/issues/13
