func.func @shift_left_tensor_ub(%t: tensor<5xi32>) -> tensor<5xi32> {
  %amnt = arith.constant dense<0xf0f0f0f0> : tensor<5xi32>
  %x = arith.shli %t, %amnt: tensor<5xi32>
  return %x: tensor<5xi32>
}
