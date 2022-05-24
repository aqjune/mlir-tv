// VERIFY
// ARGS: -memref-inputs-simple
// SKIP-IDCHECK

func.func @f(%arg: memref<1x1xf32, affine_map<(d0, d1) -> (d0 * 4 + d1 + 16)>>) {
  %ts1 = arith.constant dense<1.0>: tensor<1x1xf32>
  %ts2 = arith.constant dense<2.0>: tensor<1x1xf32>
  memref.tensor_store %ts1, %arg: memref<1x1xf32, affine_map<(d0, d1) -> (d0 * 4 + d1 + 16)>>
  memref.tensor_store %ts2, %arg: memref<1x1xf32, affine_map<(d0, d1) -> (d0 * 4 + d1 + 16)>>
  return
}
