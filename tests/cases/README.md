# Before you add new test cases...

## Directory name convention
### `<transformation-pass-name>`
ex) `fold-tensor-extract-op`
Just strip the <case_name> and extensions away, and you get the directory name.

## Naming convention
### `<transformation-pass-name>[_<case_name>].(src|tgt).mlir`
ex) `conv-to-img2col.src.mlir`, `fold-memref-subview-op_const.tgt.mlir`.
Also, each test case should form a pair of `.src.mlir` and `.tgt.mlir`.

## And some more
Each `.src.mlir` file must include the command used to create the pair `.tgt.mlir` file. For now they are written in the first-line comment. The `iree-opt` flag sometimes isn't identical to the pass name, so reproducing the src-tgt pair becomes troublesome. Including the flag(command) will be very much appreciated for this reason ;)

## Reference
https://github.com/aqjune/iree-tv/issues/13
