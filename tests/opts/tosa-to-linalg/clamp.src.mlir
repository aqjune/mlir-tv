// VERIFY

func.func @test(%arg0: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = "tosa.clamp"(%arg0) {min_int = -127 : i64, max_int = 126 : i64, min_fp = 0.0 : f32, max_fp = 0.0 : f32} : (tensor<?x?xi32>) -> tensor<?x?xi32>
	return %0: tensor<?x?xi32>
}

func.func @test2(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = "tosa.clamp"(%arg0) {min_int = -127 : i64, max_int = 126 : i64, min_fp = 1.0 : f32, max_fp = 2.0 : f32} : (tensor<?x?xf32>) -> tensor<?x?xf32>
	return %0: tensor<?x?xf32>
}
