// VERIFY

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @f(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>,
    %arg2: tensor<?x?xf32>, %arg3:tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %t1, %t2 = linalg.generic {
      indexing_maps = [#map0, #map0, #map0, #map0],
      iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1: tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%arg2, %arg3: tensor<?x?xf32>, tensor<?x?xf32>) {
  ^bb0(%x: f32, %y: f32, %dummy1: f32, %dummy2: f32):
    linalg.yield %x, %y : f32, f32
  } -> (tensor<?x?xf32>, tensor<?x?xf32>)
  return %t1, %t2: tensor<?x?xf32>, tensor<?x?xf32>
}

// mlir-opt -canonicalize
