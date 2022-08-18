#map = affine_map<(d0) -> (d0)>
module  {
  func @test_abs(%arg0: tensor<?xf32>) -> tensor<?xf32> {
		%c0 = arith.constant 0: index
    %sz = tensor.dim %arg0, %c0: tensor<?xf32>
    %0 = linalg.init_tensor [%sz] : tensor<?xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
			ins(%arg0 : tensor<?xf32>) outs(%0 : tensor<?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
      %2 = math.absf %arg1 : f32
      linalg.yield %2 : f32
    } -> tensor<?xf32>
    return %1 : tensor<?xf32>
  }
}

