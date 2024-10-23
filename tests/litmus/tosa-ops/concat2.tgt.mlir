// VERIFY

func.func @f(%t1: tensor<?xf32>, %t2: tensor<?xf32>, %t3: tensor<?xf32>) -> tensor<?xf32> {
  %tmp = "tosa.concat"(%t1, %t2) { axis = 0 : i32}
    : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %res = "tosa.concat"(%tmp, %t3) { axis = 0 : i32}
    : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %res: tensor<?xf32>
}
