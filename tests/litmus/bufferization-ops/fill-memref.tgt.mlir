// VERIFY

func.func @bufferize_fill(%arg0: memref<?xf32>) -> tensor<?xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    linalg.fill ins(%cst: f32) outs(%arg0: memref<?xf32>)

    %zerotensor = bufferization.to_tensor %arg0 : memref<?xf32> to tensor<?xf32>
    %zerotensor2 = linalg.fill ins(%cst: f32) outs(%zerotensor: tensor<?xf32>) -> tensor<?xf32>
    return %zerotensor2 : tensor<?xf32>
}
