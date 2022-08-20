// VERIFY-INCORRECT

func @f(%t: tensor<?x?xf32>, %pad_value: f32) -> tensor<?x?xf32>{
  %res = linalg.pad_tensor %t low[1, 2] high[2, 3] {
  ^bb0(%arg0 : index, %arg1 : index):
    linalg.yield %pad_value : f32
  } : tensor<?x?xf32> to tensor<?x?xf32>
  return %res: tensor<?x?xf32>
}
