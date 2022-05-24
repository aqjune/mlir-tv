func.func @f() -> (i32, i32) {
	%two = arith.constant 2: i32
	%three = arith.constant 3: i32
	return %two, %three: i32, i32
}
