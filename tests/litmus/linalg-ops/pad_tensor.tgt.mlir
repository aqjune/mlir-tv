func @f(%t: tensor<?x?xf32>, %pad_value: f32) -> (f32, f32, f32, f32, f32) {
  return %pad_value, %pad_value, %pad_value, %pad_value, %pad_value: f32,f32,f32,f32,f32
}
