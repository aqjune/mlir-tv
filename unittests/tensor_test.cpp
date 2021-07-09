#include "gtest/gtest.h"
#include "src/tensor.h"
#include "z3_expects.h"

TEST(UnitIndexTest, Default) {
  EXPECT_THROW(((z3::expr)Index()).get_sort(), z3::exception);
}

TEST(UnitIndexTest, Static) {
  EXPECT_Z3_EQ((z3::expr)Index::one(), (z3::expr)Index(1));
  EXPECT_Z3_EQ((z3::expr)Index::zero(), (z3::expr)Index(0));
}

TEST(UnitIndexTest, ConstantComparison) {
  Index zero(0);
  Index answer(42);
  Index zero_2(0);

  EXPECT_Z3_EQ((z3::expr)zero, (z3::expr)zero_2);
  EXPECT_Z3_NE((z3::expr)zero, (z3::expr)answer);
}

TEST(UnitIndexTest, VariableComparison) {
  Index named("hello");
  Index unnamed("anonymous");
  Index named_2("hello");
  Index answer(42);

  EXPECT_Z3_EQ((z3::expr)named, (z3::expr)named_2);
  EXPECT_Z3_NE((z3::expr)named, (z3::expr)unnamed);
  EXPECT_Z3_NE((z3::expr)named, (z3::expr)answer);
}

// TODO: test for Index.eval(z3::model)
// TEST(UnitIndexTest, Eval) {}


TEST(UnitFloatTest, Default) {
  EXPECT_THROW(((z3::expr)Index()).get_sort(), z3::exception);
}

TEST(UnitFloatTest, ConstantComparison) {
  Float zero(0.0);
  Float one(1.0);
  Float answer(42.0);
  Float zero_2(0.0);
  Float answer_2(llvm::APFloat(42.0));

  EXPECT_Z3_EQ((z3::expr)zero, (z3::expr)zero_2);
  EXPECT_Z3_NE((z3::expr)one, (z3::expr)answer);
  EXPECT_Z3_EQ((z3::expr)answer, (z3::expr)answer_2);
}

TEST(UnitFloatTest, VariableComparison) {
  Float named("hello");
  Float unnamed("anonymous");
  Float named_2("hello");
  Float answer(42.0);

  EXPECT_Z3_EQ((z3::expr)named, (z3::expr)named_2);
  EXPECT_Z3_NE((z3::expr)named, (z3::expr)unnamed);
  EXPECT_Z3_NE((z3::expr)named, (z3::expr)answer);
}


TEST(UnitIntegerTest, VariableComparison) {
  Integer named("hello", 32);
  Integer unnamed("anonymous", 32);
  Integer named_2("hello", 32);

  EXPECT_Z3_EQ((z3::expr)named, (z3::expr)named_2);
  EXPECT_Z3_NE((z3::expr)named, (z3::expr)unnamed);
}

TEST(UnitTensorTest, Default) {
  EXPECT_THROW(((z3::expr)Tensor()).get_sort(), z3::exception);
}
