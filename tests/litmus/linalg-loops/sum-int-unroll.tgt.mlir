func @sum() -> tensor<i8>
{
  %fifty = arith.constant 50: i8
  %t = linalg.init_tensor []: tensor<i8>
  %t2 = tensor.insert %fifty into %t[]: tensor<i8>
  return %t2: tensor<i8>
}
