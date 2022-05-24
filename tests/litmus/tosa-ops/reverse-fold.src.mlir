// VERIFY
// ARGS: -max-unknown-dimsize=25

func.func @f(%t: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %rt = "tosa.reverse"(%t) {axis = 1 : i64} : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %rt2 = "tosa.reverse"(%rt) {axis = 1 : i64} : (tensor<?x?xf32>) -> tensor<?x?xf32>
  return %rt2: tensor<?x?xf32>
}
