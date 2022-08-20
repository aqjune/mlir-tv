// VERIFY

func @select(%arg0: i1, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32> {
  %0 = select %arg0, %arg1, %arg2 : tensor<?xf32>
  return %0 : tensor<?xf32>
}
