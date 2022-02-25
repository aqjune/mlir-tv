# Before you add new test cases...

## Directory name convention
### Opts: `<transformation-pass-name>`
ex) `opts/fold-tensor-extract-op`
### Litmus: `<feature-category-name>`
ex) `litmus/fp-ops`

## File name convention
### `<case_name>[-bad].(src|tgt).mlir`
ex) `nhwc_filter.src.mlir`, `const_tensor.tgt.mlir`, `i32-bad.src.mlir`.  
Also, each test case should form a pair of `.src.mlir` and `.tgt.mlir`.  
Use suffix `-bad` for `// VERIFY-INCORRECT` test cases

## Test keywords
`// VERIFY` : Check if the transformation is correct  
`// VERIFY-INCORRECT` : Check if the transformation is indeed wrong  
`// UNSUPPORTED` : Use to ignore the test case that includes **yet** unimplemented dialects  
`// EXPECT: "<message>"[ ((&& "message")+|(|| "message")+)]` : Check if the stdout/stderr includes the provided message(s).
* If messages are delimited by &&, the test passes iff every message is included
* If messages are delimited by ||, the test passes if at least one of them is included.
* **Using both && and || is not allowed.**

## Test options
`// ARGS: <arg>[( <arg>)+]` : Pass given arguments to `mlir-tv` for given test, except for identity checks.  
`// SKIP-IDCHECK` : Skip identity checks for given test. **Cannot be used with `ARGS-IDCHECK`!**  
`// ARGS-IDCHECK: <arg>[( <arg>)+]` : Pass given arguments to `mlir-tv` when running identity checks for given test. **Cannot be used with `SKIP-IDCHECK`!**

## Writing keywords and options
All `src.mlir` must start with test keyword or test option. They must include one and only test keyword, and may include one or more test options.   
Each keyword and option must be written consecutively until there's no more to use.  
Then, the codes should appear after an empty line.

## And some more
Each `.src.mlir` file must include the command used to create the pair `.tgt.mlir` file. For now they are written in the first-line comment. The `iree-opt` flag sometimes isn't identical to the pass name, so reproducing the src-tgt pair becomes troublesome. Including the flag(command) will be very much appreciated for this reason ;)  
When implementing a negative test case by modifying valid `tgt.mlir`, the diff must be annotated with comments.

## Reference
https://github.com/aqjune/mlir-tv/issues/13
