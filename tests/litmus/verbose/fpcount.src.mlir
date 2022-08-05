// ARGS: --verbose
// EXPECT: "f32 arg count: 2"

// [analysis]:   f32 arg count: 2
// [analysis]:   f32 var count: 0
// [analysis]:   f32 element counts: 9998
func @f(%a: tensor<50x100xf32>, %b: tensor<50x100xf32>) -> () {
  return
}
