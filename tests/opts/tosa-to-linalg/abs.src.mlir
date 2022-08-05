// VERIFY

func @test_abs(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "tosa.abs"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

