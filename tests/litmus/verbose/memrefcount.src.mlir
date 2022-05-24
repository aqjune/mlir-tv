// ARGS: --verbose
// EXPECT: "memref arg count (f32): 2"
func.func @f(%a: memref<50x100xf32>, %b: memref<50x100xf32>) -> () {
  return
}
