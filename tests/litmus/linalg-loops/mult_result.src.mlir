// VERIFY

#accesses = [
  affine_map<(i) -> (i)>,
  affine_map<(i) -> (i)>,
  affine_map<(i) -> (i)>
]
#trait = {
  indexing_maps = #accesses,
  iterator_types = ["parallel"]
}
func @linalg_op_same_out_tensors(
    %t1: tensor<?xf32> {linalg.inplaceable = true},
    %t2: tensor<?xf32> {linalg.inplaceable = true},
    %t3: tensor<?xf32> {linalg.inplaceable = true}) -> (tensor<?xf32>, tensor<?xf32>){
  %o:2 = linalg.generic #trait ins(%t1 : tensor<?xf32>)
                               outs (%t2, %t3 : tensor<?xf32>, tensor<?xf32>) {
      ^bb(%0: f32, %1: f32, %2 : f32) :
        linalg.yield %0, %1 : f32, f32
    } -> (tensor<?xf32>, tensor<?xf32>)
  return %o#0, %o#1 : tensor<?xf32>, tensor<?xf32>
}

// How to reproduce tgt:
// mlir-opt -linalg-comprehensive-module-bufferize=test-analysis-only <src>