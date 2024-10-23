// VERIFY

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1 : dense, d0 : compressed)
}>

func.func @f(%x: tensor<?x?xf32>) -> tensor<?x?xf32>
{
  %y = sparse_tensor.convert %x: tensor<?x?xf32> to tensor<?x?xf32, #CSR>
  %z = sparse_tensor.convert %y: tensor<?x?xf32, #CSR> to tensor<?x?xf32>
  return %x: tensor<?x?xf32> 
}
