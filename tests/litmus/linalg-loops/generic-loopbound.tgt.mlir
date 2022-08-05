#accesses = [
  affine_map<(m) -> (m)>,
  affine_map<(m) -> (m)>,
  affine_map<(m) -> (m)>
]

#attrs = {
  indexing_maps = #accesses,
  iterator_types = ["parallel"]
}

func @example(%A: tensor<4xf32>,
              %B: tensor<?xf32>,
              %out_shape: tensor<?xf32>) {
  %tmp = linalg.generic #attrs
  ins(%B, %A: tensor<?xf32>, tensor<4xf32>) // input args' order swapped
  outs(%out_shape: tensor<?xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
		linalg.yield %b: f32
  } -> tensor<?xf32>
  return
}
