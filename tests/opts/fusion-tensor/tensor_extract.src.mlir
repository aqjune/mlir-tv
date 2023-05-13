// VERIFY

func.func @sigmoid_dynamic_dim(%0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  %cp5 = arith.constant 5.000000e-01 : f32
  %c0 = arith.constant 0 : index
  %shape = shape.shape_of %0 : tensor<?x1xf32> -> tensor<?xindex>
  %extend = shape.to_extent_tensor %shape : tensor<?xindex> -> tensor<2xindex>
  %extracted = tensor.extract %extend[%c0] : tensor<2xindex>
  %init0 = tensor.empty (%extracted) : tensor<?x1xf32>
  %1 = linalg.generic {indexing_maps = [
    affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  }
     outs(%init0 : tensor<?x1xf32>) {
    ^bb0(%a: f32):  // no predecessors
      linalg.yield %cp5 : f32
  } -> tensor<?x1xf32>
  %d0 = tensor.dim %0, %c0 : tensor<?x1xf32>
  %init1 = tensor.empty (%d0) : tensor<?x1xf32>
  %2 = linalg.generic {indexing_maps = [
    affine_map<(d0, d1) -> (d0, d1)>,
    affine_map<(d0, d1) -> (d0, d1)>,
    affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  }
      ins(%0, %1 : tensor<?x1xf32>, tensor<?x1xf32>)
     outs(%init1 : tensor<?x1xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):  // no predecessors
      %m = arith.mulf %a, %b : f32
      linalg.yield %m : f32
  } -> tensor<?x1xf32>
  return %2 : tensor<?x1xf32>
}

// How to reproduce tgt:
// iree-opt -linalg-fusion-for-tensor-ops <src>
