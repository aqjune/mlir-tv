// VERIFY
// ARGS: --use-fn-argument-dims=dyn_tensor_1@1,dyn_tensor_2@4

func.func private @dyn_tensor_1(%v1: f32, %v: tensor<3x?x?xf32>) -> tensor<3x?x?xf32>
func.func private @dyn_tensor_2(%v1: f32, %v2: f32, %v3: f32, %v4: f32, %v: tensor<3x?x?xf32>) -> tensor<3x?x?xf32>

func.func @tensor_dim(%v: tensor<3x?x?xf32>) -> i1 {
  %f0 = arith.constant 1.0: f32
  %t1 = func.call @dyn_tensor_1(%f0, %v): (f32, tensor<3x?x?xf32>) -> tensor<3x?x?xf32>
  %t2 = func.call @dyn_tensor_2(%f0, %f0, %f0, %f0, %v): (f32, f32, f32, f32, tensor<3x?x?xf32>) -> tensor<3x?x?xf32>
  %c0 = arith.constant 0: index
  %d1 = tensor.dim %t1, %c0: tensor<3x?x?xf32>
  %d2 = tensor.dim %t2, %c0: tensor<3x?x?xf32>
  %r = arith.cmpi eq, %d1, %d2 : index
  return %r: i1
}
