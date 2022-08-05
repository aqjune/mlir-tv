// VERIFY

func @f(%t: tensor<?x?x?xi32>) -> tensor<?x?x?xi32> {
  %rt = "tosa.reverse"(%t) {axis = 1 : i64} : (tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  return %rt: tensor<?x?x?xi32>
}

// mlir-opt -tosa-to-linalg
