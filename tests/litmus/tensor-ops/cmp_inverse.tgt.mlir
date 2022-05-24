func.func @ole(%arg0: tensor<5x3xf32>, %arg1: tensor<5x3xf32>) -> tensor<5x3xi1> {
  %c = arith.cmpf "oge", %arg1, %arg0 : tensor<5x3xf32>
  return %c : tensor<5x3xi1>
}

func.func @olt(%arg0: tensor<5x3xf32>, %arg1: tensor<5x3xf32>) -> tensor<5x3xi1> {
  %c = arith.cmpf "ogt", %arg1, %arg0 : tensor<5x3xf32>
  return %c : tensor<5x3xi1>
}

func.func @oge(%arg0: tensor<5x3xf32>, %arg1: tensor<5x3xf32>) -> tensor<5x3xi1> {
  %c = arith.cmpf "ole", %arg1, %arg0 : tensor<5x3xf32>
  return %c : tensor<5x3xi1>
}

func.func @ogt(%arg0: tensor<5x3xf32>, %arg1: tensor<5x3xf32>) -> tensor<5x3xi1> {
  %c = arith.cmpf "olt", %arg1, %arg0 : tensor<5x3xf32>
  return %c : tensor<5x3xi1>
}

func.func @ule(%arg0: tensor<5x3xf32>, %arg1: tensor<5x3xf32>) -> tensor<5x3xi1> {
  %c = arith.cmpf "uge", %arg1, %arg0 : tensor<5x3xf32>
  return %c : tensor<5x3xi1>
}

func.func @ult(%arg0: tensor<5x3xf32>, %arg1: tensor<5x3xf32>) -> tensor<5x3xi1> {
  %c = arith.cmpf "ugt", %arg1, %arg0 : tensor<5x3xf32>
  return %c : tensor<5x3xi1>
}

func.func @uge(%arg0: tensor<5x3xf32>, %arg1: tensor<5x3xf32>) -> tensor<5x3xi1> {
  %c = arith.cmpf "ule", %arg1, %arg0 : tensor<5x3xf32>
  return %c : tensor<5x3xi1>
}

func.func @ugt(%arg0: tensor<5x3xf32>, %arg1: tensor<5x3xf32>) -> tensor<5x3xi1> {
  %c = arith.cmpf "ult", %arg1, %arg0 : tensor<5x3xf32>
  return %c : tensor<5x3xi1>
}
