// VERIFY

func.func @add(%arg0: tensor<2x1xf32>, %arg1: tensor<1x3xf32>) -> tensor<2x3xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %x11 = tensor.extract %arg0[%c0, %c0] : tensor<2x1xf32>
  %x21 = tensor.extract %arg0[%c1, %c0] : tensor<2x1xf32>
  %y11 = tensor.extract %arg1[%c0, %c0] : tensor<1x3xf32>
  %y12 = tensor.extract %arg1[%c0, %c1] : tensor<1x3xf32>
  %y13 = tensor.extract %arg1[%c0, %c2] : tensor<1x3xf32>
  %tx0 = linalg.init_tensor [2, 3] : tensor<2x3xf32>
  %ty0 = linalg.init_tensor [2, 3] : tensor<2x3xf32>
  %tx1 = tensor.insert %x11 into %tx0[%c0, %c0] : tensor<2x3xf32>
  %tx2 = tensor.insert %x11 into %tx1[%c0, %c1] : tensor<2x3xf32>
  %tx3 = tensor.insert %x11 into %tx2[%c0, %c2] : tensor<2x3xf32>
  %tx4 = tensor.insert %x21 into %tx3[%c1, %c0] : tensor<2x3xf32>
  %tx5 = tensor.insert %x21 into %tx4[%c1, %c1] : tensor<2x3xf32>
  %tx6 = tensor.insert %x21 into %tx5[%c1, %c2] : tensor<2x3xf32>

  %ty1 = tensor.insert %y11 into %ty0[%c0, %c0] : tensor<2x3xf32>
  %ty2 = tensor.insert %y12 into %ty1[%c0, %c1] : tensor<2x3xf32>
  %ty3 = tensor.insert %y13 into %ty2[%c0, %c2] : tensor<2x3xf32>
  %ty4 = tensor.insert %y11 into %ty3[%c1, %c0] : tensor<2x3xf32>
  %ty5 = tensor.insert %y12 into %ty4[%c1, %c1] : tensor<2x3xf32>
  %ty6 = tensor.insert %y13 into %ty5[%c1, %c2] : tensor<2x3xf32>

  %0 = "tosa.add"(%tx6, %ty6) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}