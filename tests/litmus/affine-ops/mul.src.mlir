// VERIFY

func.func @f(%x: index) -> index {
  %res = affine.apply affine_map<(i) -> (i*2)> (%x)
  return %res: index
}
