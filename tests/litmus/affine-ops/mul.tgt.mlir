func.func @f(%x: index) -> index {
  %res = affine.apply affine_map<(i) -> (i+i)> (%x)
  return %res: index
}
