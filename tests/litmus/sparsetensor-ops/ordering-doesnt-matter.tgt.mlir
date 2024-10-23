// VERIFY

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1: dense, d0: compressed)
}>

func.func @f(%x: tensor<?x?xf32>) -> f32
{
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %res = tensor.extract %x[%c0, %c1]: tensor<?x?xf32>
  return %res: f32 
}
