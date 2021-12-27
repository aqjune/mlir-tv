// VERIFY
// ARGS: --unroll-int-sum

func @sum() -> tensor<i8>
{
  %cst = arith.constant dense<10> : tensor<5xi8>
  %zero = arith.constant 0 : i8
  %i = linalg.init_tensor []: tensor<i8>
  %outty = linalg.fill(%zero, %i) : i8, tensor<i8> -> tensor<i8>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]}
     ins(%cst : tensor<5xi8>) outs(%outty : tensor<i8>) {
     ^bb0(%arg0 : i8, %arg1 : i8):
        %0 = arith.addi %arg0, %arg1 : i8
        linalg.yield %0 : i8
  } -> tensor<i8>
  return %result : tensor<i8>
}
