func.func @sum() -> tensor<i8>
{
  %fifty = arith.constant 50: i8
  %zero = arith.constant 0 : i8
  %i = tensor.empty (): tensor<i8>
  %t = linalg.fill ins(%zero: i8) outs(%i: tensor<i8>) -> tensor<i8>
  %t2 = tensor.insert %fifty into %t[]: tensor<i8>
  return %t2: tensor<i8>
}
