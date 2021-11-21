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
func @generic_mult2(
    %t1: tensor<5xf32>,
    %t2: tensor<5xf32>,
    %t3: tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>){
  %o:2 = linalg.generic #trait ins(%t1 : tensor<5xf32>)
                               outs (%t2, %t3 : tensor<5xf32>, tensor<5xf32>) {
      ^bb(%0: f32, %1: f32, %2 : f32) :
        linalg.yield %0, %2 : f32, f32
    } -> (tensor<5xf32>, tensor<5xf32>)
  return %o#0, %o#1 : tensor<5xf32>, tensor<5xf32>
}