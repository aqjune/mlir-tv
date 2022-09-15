// VERIFY
// ARGS: --use-fn-argument-dims=dyn_tensor_1@1,dyn_tensor_2@4

func.func private @dyn_tensor_1(%v1: f32, %v: tensor<3x?x?xf32>) -> tensor<3x?x?xf32>
func.func private @dyn_tensor_2(%v1: f32, %v2: f32, %v3: f32, %v4: f32, %v: tensor<3x?x?xf32>) -> tensor<3x?x?xf32>

func.func @tensor_dim(%v: tensor<3x?x?xf32>) -> index {
  %f0 = arith.constant 1.0: f32
  %t1 = func.call @dyn_tensor_1(%f0, %v): (f32, tensor<3x?x?xf32>) -> tensor<3x?x?xf32>
  %c0 = arith.constant 1: index
  %r = tensor.dim %t1, %c0: tensor<3x?x?xf32>
  return %r: index
}
