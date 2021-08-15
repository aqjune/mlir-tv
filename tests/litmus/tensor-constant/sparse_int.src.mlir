// VERIFY

func @f() -> i32 {
  %const_two = constant 2: index
  %const_zero = constant 0: index
  %c = constant sparse<[[0, 0], [1, 0], [2, 0]],  [-1, -2, -3]> : tensor<5x1xi32>
  %minus_three = tensor.extract %c[%const_two, %const_zero] : tensor<5x1xi32>
  return %minus_three: i32
}
