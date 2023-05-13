// VERIFY

#map = affine_map<(d0) -> (d0)>
func.func @dumb_loop(%arg0: tensor<?xi32>) -> tensor<?xi32> {
  %c0 = arith.constant 0: index
  %sz = tensor.dim %arg0, %c0: tensor<?xi32>
  %outty = tensor.empty (%sz) : tensor<?xi32>

  %res = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%arg0: tensor<?xi32>)
      outs(%outty : tensor<?xi32>) {
  ^bb0(%a: i32, %dummy: i32):

    // This loop just returns %arg0.
    %res0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
        ins(%arg0: tensor<?xi32>)
        outs(%outty : tensor<?xi32>) {
    ^bb0(%b: i32, %dummy2: i32):
      linalg.yield %b: i32
    } -> tensor<?xi32>

    %idx = linalg.index 0: index
    %elem = tensor.extract %res0[%idx]: tensor<?xi32>
    linalg.yield %elem : i32
  } -> tensor<?xi32>

  return %res : tensor<?xi32>
}
