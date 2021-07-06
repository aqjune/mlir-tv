# Before you add new test cases...

## File naming convention
### `<src filename(noext)>_test.cpp`
ex) `smt_test.cpp` includes the tests for `smt.cpp`.

## Test suite naming convention
### `Unit<SrcFilename(noext)><SuiteName>`
ex) `UnitSMTTest`, `UnitTensorTest`.  
**Unit** prefix is important: this distinguishes unit tests from tests for passes.  

## Test case naming convention
Name them as you like, just use `CamelCase`.
