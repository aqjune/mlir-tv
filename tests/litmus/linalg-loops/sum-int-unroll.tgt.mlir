func @sum() -> tensor<i8>
{
  %fifty = arith.constant 50: i8
  %zero = arith.constant 0 : i8
  %i = linalg.init_tensor []: tensor<i8>
  %t = linalg.fill(%zero, %i): i8, tensor<i8> -> tensor<i8>
  %t2 = tensor.insert %fifty into %t[]: tensor<i8>
  return %t2: tensor<i8>
}
