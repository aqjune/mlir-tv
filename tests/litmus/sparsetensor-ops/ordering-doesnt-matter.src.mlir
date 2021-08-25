// VERIFY

#CSR = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (j,i)>
}>

func @f(%x: tensor<?x?xf32>) -> f32
{
  %c0 = constant 0: index
  %c1 = constant 1: index
  %y = sparse_tensor.convert %x: tensor<?x?xf32> to tensor<?x?xf32, #CSR>
  %res = tensor.extract %y[%c0, %c1]: tensor<?x?xf32, #CSR>
  return %res: f32 
}
