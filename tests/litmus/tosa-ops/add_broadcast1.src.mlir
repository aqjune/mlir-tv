// VERIFY

func.func @add(%arg0: tensor<1xf32>, %arg1: tensor<10x9x8x7xf32>) -> tensor<10x9x8x7xf32> {
  %c0 = arith.constant 0 : index
  %c = tensor.extract %arg0[%c0] : tensor<1xf32>
  %t1 = tensor.empty () : tensor<10x9x8x7xf32>
  %t2 = linalg.fill ins(%c: f32) outs(%t1: tensor<10x9x8x7xf32>) -> tensor<10x9x8x7xf32>
  %0 = "tosa.add"(%t2, %arg1) : (tensor<10x9x8x7xf32>, tensor<10x9x8x7xf32>) -> tensor<10x9x8x7xf32>
  
  return %0 : tensor<10x9x8x7xf32>
}
