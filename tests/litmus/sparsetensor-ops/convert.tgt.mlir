#CSR = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>
}>

func.func @f(%x: tensor<?x?xf32>) -> tensor<?x?xf32>
{
  return %x: tensor<?x?xf32> 
}