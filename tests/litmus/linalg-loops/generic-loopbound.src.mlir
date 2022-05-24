// VERIFY

// This is a test to show how our encoding calculates the loop bounds.
// It uses the first matched tensor's dimension and this should be matched with other tensor's dimension (otherwise it's UB)

#accesses = [
  affine_map<(m) -> (m)>,
  affine_map<(m) -> (m)>,
  affine_map<(m) -> (m)>
]

#attrs = {
  indexing_maps = #accesses,
  iterator_types = ["parallel"]
}

func.func @example(%A: tensor<4xf32>,
              %B: tensor<?xf32>,
              %out_shape: tensor<?xf32>) {
  %tmp = linalg.generic #attrs
  ins(%A, %B: tensor<4xf32>, tensor<?xf32>) outs(%out_shape: tensor<?xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
		linalg.yield %a: f32
  } -> tensor<?xf32>
  return
}
