// VERIFY

#CSR = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>
}>

func.func @f(%x: tensor<?x?xf32>) -> tensor<?x?xf32>
{
  %y = sparse_tensor.convert %x: tensor<?x?xf32> to tensor<?x?xf32, #CSR>
  %z = sparse_tensor.convert %y: tensor<?x?xf32, #CSR> to tensor<?x?xf32>
  return %x: tensor<?x?xf32> 
}
