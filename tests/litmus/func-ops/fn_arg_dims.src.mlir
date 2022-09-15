// VERIFY
// ARGS: --use-fn-argument-dims=dyn_tensor_1@0

func.func private @dyn_tensor_1(%v: tensor<3x?x?xf32>) -> tensor<3x?x?xf32>

func.func @tensor_dim(%v: tensor<3x?x?xf32>) -> index {
  %t = func.call @dyn_tensor_1(%v): (tensor<3x?x?xf32>) -> tensor<3x?x?xf32>
  %c = arith.constant 1: index
  %r = tensor.dim %v, %c: tensor<3x?x?xf32>
  return %r: index
}
