#define EXPECT_Z3_EQ(lhs, rhs) EXPECT_EQ((Z3_ast)lhs, (Z3_ast)rhs)
#define EXPECT_Z3_NE(lhs, rhs) EXPECT_NE((Z3_ast)lhs, (Z3_ast)rhs)

#define ZE smt::expr
#define ZE_INDEX (ZE)Index
#define ZE_INTEGER (ZE)Integer
#define ZE_FLOAT (ZE)Float
