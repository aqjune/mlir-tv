// VERIFY

func.func @f(%t: tensor<?x?x?xi32>) -> tensor<?x?x?xi32> {
  %rt = "tosa.reverse"(%t) {axis = 1 : i32} : (tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  return %rt: tensor<?x?x?xi32>
}

// mlir-opt -tosa-to-linalg
