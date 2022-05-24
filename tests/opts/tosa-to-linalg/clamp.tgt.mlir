#map = affine_map<(d0, d1) -> (d0, d1)>
module  {
  func.func @test(%arg0: tensor<?x?xi32>) -> tensor<?x?xi32> {
    %c0 = arith.constant 0 : index
    %0 = tensor.dim %arg0, %c0 : tensor<?x?xi32>
    %c1 = arith.constant 1 : index
    %1 = tensor.dim %arg0, %c1 : tensor<?x?xi32>
    %2 = linalg.init_tensor [%0, %1] : tensor<?x?xi32>
    %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<?x?xi32>) outs(%2 : tensor<?x?xi32>) {
    ^bb0(%arg1: i32, %arg2: i32):  // no predecessors
      %c-127_i32 = arith.constant -127 : i32
      %c126_i32 = arith.constant 126 : i32
      %4 = arith.cmpi slt, %arg1, %c-127_i32 : i32
      %5 = arith.select %4, %c-127_i32, %arg1 : i32
      %6 = arith.cmpi slt, %c126_i32, %arg1 : i32
      %7 = arith.select %6, %c126_i32, %5 : i32
      linalg.yield %7 : i32
    } -> tensor<?x?xi32>
    return %3 : tensor<?x?xi32>
  }
  func.func @test2(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %c1 = arith.constant 1 : index
    %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %2 = linalg.init_tensor [%0, %1] : tensor<?x?xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<?x?xf32>) outs(%2 : tensor<?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
      %cst = arith.constant 1.000000e+00 : f32
      %cst_0 = arith.constant 2.000000e+00 : f32
      %4 = arith.cmpf olt, %arg1, %cst : f32
      %5 = arith.select %4, %cst, %arg1 : f32
      %6 = arith.cmpf olt, %cst_0, %arg1 : f32
      %7 = arith.select %6, %cst_0, %5 : f32
      linalg.yield %7 : f32
    } -> tensor<?x?xf32>
    return %3 : tensor<?x?xf32>
  }
}

