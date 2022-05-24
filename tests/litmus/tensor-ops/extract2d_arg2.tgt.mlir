func.func @extract(%v: tensor<?x?xf32>, %cx: index, %cy: index) -> tensor<?x?xf32>
{
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %c2 = arith.constant 2: index
  %s = tensor.extract_slice %v[%cy,0][%c2,%cx][1,%c1]: tensor<?x?xf32> to tensor<?x?xf32>
  return %s: tensor<?x?xf32>
}