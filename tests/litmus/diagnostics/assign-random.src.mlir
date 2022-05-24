// ARGS: -assign-random-to-unsupported-ops
// EXPECT: "Assigning any value to this op (linalg.batch_matvec).."
// SKIP-IDCHECK

func.func @f(%ta: tensor<?x?x?xf32>, %tb: tensor<?x?x?xf32>, %tc: tensor<?x?x?xf32>,
		%tv: tensor<?x?xf32>, %tv2: tensor<?x?xf32>) {
  // The first unusupported operation (ATM)
  %res1 = linalg.batch_matmul ins(%ta, %tb: tensor<?x?x?xf32>, tensor<?x?x?xf32>)
		outs(%tc: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  // The second unsupported operation (ATM)
  %res2 = linalg.batch_matvec ins(%ta, %tv: tensor<?x?x?xf32>, tensor<?x?xf32>)
		outs(%tv2: tensor<?x?xf32>) -> tensor<?x?xf32>
	return
}
