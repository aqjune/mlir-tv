// VERIFY
// ARGS: --use-fn-argument-dims=dyn_tensor_1@0

func.func private @dyn_tensor_1(%v: tensor<3x?x?xf32>) -> tensor<3x?x?xf32>

func.func @tensor_dim(%v: tensor<3x?x?xf32>) -> i1 {
  %t = func.call @dyn_tensor_1(%v): (tensor<3x?x?xf32>) -> tensor<3x?x?xf32>
  %c0 = arith.constant 0: index
  %d1 = tensor.dim %v, %c0: tensor<3x?x?xf32>
  %d2 = tensor.dim %t, %c0: tensor<3x?x?xf32>
  %r = arith.cmpi eq, %d1, %d2 : index
  return %r: i1
}
