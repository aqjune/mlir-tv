// ARGS: --verbose
// EXPECT: "f32 arg count: 10000"
func @f(%a: tensor<50x100xf32>, %b: tensor<50x100xf32>) -> () {
  return
}
