// EXPECT: "float: limit bits: 0, smaller value bits: 6, precision bits: 0"
// ARGS: -fp-bits=5 -verbose

// 6 = 5 + 1(sign bit)

func.func @f() {
  return
}
