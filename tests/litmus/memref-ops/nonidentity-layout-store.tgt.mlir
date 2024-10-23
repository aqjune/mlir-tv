func.func @f(%arg: memref<1x1xf32, affine_map<(d0, d1) -> (d0 * 4 + d1 + 16)>>) {
  %one = arith.constant 1.0:f32
  %two = arith.constant 2.0:f32
  //linalg.fill ins(%one:f32) outs(%arg: memref<1x1xf32, affine_map<(d0, d1) -> (d0 * 4 + d1 + 16)>>)
  linalg.fill ins(%two:f32) outs(%arg: memref<1x1xf32, affine_map<(d0, d1) -> (d0 * 4 + d1 + 16)>>)
  return
}
