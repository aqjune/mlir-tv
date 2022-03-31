
memref.global "private" constant @constant_a : memref<8x4xf32> = dense<1.0>
memref.global "private" constant @constant_b : memref<4x16xf32> = dense<2.0>
func @f() -> tensor<8x16xf32> {
  %a = memref.get_global @constant_a : memref<8x4xf32>
  %b = memref.get_global @constant_b : memref<4x16xf32>
  %ta = bufferization.to_tensor %a : memref<8x4xf32>
  %tb = bufferization.to_tensor %b : memref<4x16xf32>
  %c = linalg.init_tensor [8, 16] : tensor<8x16xf32>
  %cst = arith.constant -0.0 : f32
  %tc = linalg.fill ins(%cst: f32) outs(%c: tensor<8x16xf32>) -> tensor<8x16xf32>
  %mat = linalg.matmul ins(%ta, %tb: tensor<8x4xf32>, tensor<4x16xf32>) outs(%tc: tensor<8x16xf32>) -> tensor<8x16xf32>
  return %mat : tensor<8x16xf32>
}
