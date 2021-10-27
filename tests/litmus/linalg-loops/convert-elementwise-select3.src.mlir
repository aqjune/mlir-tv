// VERIFY

func @select(%arg0: tensor<?x?xi1>, %arg1: tensor<?x?xi32>, %arg2: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = select %arg0, %arg1, %arg2 : tensor<?x?xi1>, tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}

// How to reproduce tgt:
// mlir-opt -convert-elementwise-to-linalg <src>
