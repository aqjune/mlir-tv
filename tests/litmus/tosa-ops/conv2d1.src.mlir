// VERIFY

func.func @conv(%arg0: tensor<1x16x16x4xf32>, %arg1: tensor<3x3x4x16xf32>) -> tensor<1x14x14x16xf32> {
    %c0 = arith.constant -0.0 : f32
    %bias = tensor.from_elements %c0,%c0,%c0,%c0,%c0,%c0,%c0,%c0,%c0,%c0,%c0,%c0,%c0,%c0,%c0,%c0: tensor<16xf32>
    %filperms = "tosa.const"() {value = dense<[3, 0, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
    %arg3 = "tosa.transpose"(%arg1, %filperms) : (tensor<3x3x4x16xf32>, tensor<4xi32>) -> tensor<16x3x3x4xf32>
    %0 = "tosa.conv2d"(%arg0, %arg3, %bias) {dilation = array<i64:1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x16x16x4xf32>, tensor<16x3x3x4xf32>, tensor<16xf32>) -> tensor<1x14x14x16xf32>
    return %0 : tensor<1x14x14x16xf32>
}
