#include "gtest/gtest.h"
#include "src/value.h"
#include "tv_test_shared.h"

TEST(UnitIndexTest, Default) {
  EXPECT_THROW((ZE_INDEX()).get_sort(), z3::exception);
}

TEST(UnitIndexTest, Static) {
  EXPECT_Z3_EQ(ZE_INDEX::one(), ZE_INDEX(1));
  EXPECT_Z3_EQ(ZE_INDEX::zero(), ZE_INDEX(0));
}

TEST(UnitIndexTest, ConstantComparison) {
  Index zero(0);
  Index answer(42);
  Index zero_2(0);

  EXPECT_Z3_EQ((ZE)zero, (ZE)zero_2);
  EXPECT_Z3_NE((ZE)zero, (ZE)answer);
}

TEST(UnitIndexTest, VariableComparison) {
  Index named("hello");
  Index unnamed("anonymous");
  Index named_2("hello");
  Index answer(42);

  EXPECT_Z3_EQ((ZE)named, (ZE)named_2);
  EXPECT_Z3_NE((ZE)named, (ZE)unnamed);
  EXPECT_Z3_NE((ZE)named, (ZE)answer);
}

// TODO: test for Index.eval(z3::model)
// TEST(UnitIndexTest, Eval) {}


TEST(UnitFloatTest, Default) {
  EXPECT_THROW((ZE_INDEX()).get_sort(), z3::exception);
}

TEST(UnitFloatTest, ConstantComparison) {
  Float zero(0.0);
  Float one(1.0);
  Float answer(42.0);
  Float zero_2(0.0);
  Float answer_2(llvm::APFloat(42.0));

  EXPECT_Z3_EQ((ZE)zero, (ZE)zero_2);
  EXPECT_Z3_NE((ZE)one, (ZE)answer);
  EXPECT_Z3_EQ((ZE)answer, (ZE)answer_2);
}

TEST(UnitFloatTest, VariableComparison) {
  Float named("hello");
  Float unnamed("anonymous");
  Float named_2("hello");
  Float answer(42.0);

  EXPECT_Z3_EQ((ZE)named, (ZE)named_2);
  EXPECT_Z3_NE((ZE)named, (ZE)unnamed);
  EXPECT_Z3_NE((ZE)named, (ZE)answer);
}


TEST(UnitIntegerTest, VariableComparison) {
  Integer named("hello", 32);
  Integer unnamed("anonymous", 32);
  Integer named_2("hello", 32);

  EXPECT_Z3_EQ((ZE)named, (ZE)named_2);
  EXPECT_Z3_NE((ZE)named, (ZE)unnamed);
}


TEST(UnitTensorTest, Default) {
  EXPECT_THROW(((ZE)Tensor()).get_sort(), z3::exception);
}

TEST(UnitTensorTest, Splat1D) {
  Integer elem("elem", 32);
  Integer not_elem("eeee", 32);

  Tensor splat((ZE)elem, std::vector<ZE>{ZE_INDEX(42)});
  EXPECT_Z3_EQ((ZE)elem, splat.get(std::vector<ZE>{ZE_INDEX(0)}).simplify());
  EXPECT_Z3_EQ((ZE)elem, splat.get(std::vector<ZE>{ZE_INDEX(3)}).simplify());
  EXPECT_Z3_EQ((ZE)elem, splat.get(std::vector<ZE>{ZE_INDEX(41)}).simplify());

  EXPECT_Z3_NE((ZE)not_elem, splat.get(std::vector<ZE>{ZE_INDEX(2)}).simplify());
  EXPECT_Z3_NE((ZE)not_elem, splat.get(std::vector<ZE>{ZE_INDEX(22)}).simplify());
}

TEST(UnitTensorTest, Splat3D) {
  Integer elem("elem", 32);
  Integer not_elem("eeee", 32);

  Tensor splat((ZE)elem, std::vector<ZE>{ZE_INDEX(3), ZE_INDEX(4), ZE_INDEX(5)});
  EXPECT_Z3_EQ((ZE)elem, splat.get(std::vector<ZE>{ZE_INDEX(0), ZE_INDEX(0), ZE_INDEX(0)}).simplify());
  EXPECT_Z3_EQ((ZE)elem, splat.get(std::vector<ZE>{ZE_INDEX(1), ZE_INDEX(2), ZE_INDEX(3)}).simplify());
  EXPECT_Z3_EQ((ZE)elem, splat.get(std::vector<ZE>{ZE_INDEX(2), ZE_INDEX(2), ZE_INDEX(2)}).simplify());

  EXPECT_Z3_NE((ZE)not_elem, splat.get(std::vector<ZE>{ZE_INDEX(0), ZE_INDEX(1), ZE_INDEX(2)}).simplify());
  EXPECT_Z3_NE((ZE)not_elem, splat.get(std::vector<ZE>{ZE_INDEX(1), ZE_INDEX(1), ZE_INDEX(1)}).simplify());
}

TEST(UnitTensorTest, Elem1D) {
  Tensor tensor(std::vector<ZE>{ZE_FLOAT(1.0), ZE_FLOAT(3.0), ZE_FLOAT(5.0), ZE_FLOAT(7.0), ZE_FLOAT(9.0)});
  Float idx0(1.0);
  Float idx2(5.0);
  Float idx4(9.0);

  EXPECT_Z3_EQ((ZE)idx0, tensor.get(std::vector<ZE>{ZE_INDEX(0)}).simplify());
  EXPECT_Z3_NE((ZE)idx0, tensor.get(std::vector<ZE>{ZE_INDEX(1)}).simplify());
  EXPECT_Z3_EQ((ZE)idx2, tensor.get(std::vector<ZE>{ZE_INDEX(2)}).simplify());
  EXPECT_Z3_NE((ZE)idx0, tensor.get(std::vector<ZE>{ZE_INDEX(3)}).simplify());
  EXPECT_Z3_EQ((ZE)idx4, tensor.get(std::vector<ZE>{ZE_INDEX(4)}).simplify());
}

TEST(UnitTensorTest, Named1D) {
  Tensor named_int("named", std::vector<ZE>{ZE_INDEX(42)}, Integer::sort(32));
  EXPECT_Z3_EQ(Integer::sort(32), named_int.get(std::vector<ZE>{ZE_INDEX(0)}).simplify().get_sort());
  EXPECT_Z3_NE(Float::sort(), named_int.get(std::vector<ZE>{ZE_INDEX(0)}).simplify().get_sort());

  Tensor named_float("named", std::vector<ZE>{ZE_INDEX(42)}, Float::sort());
  EXPECT_Z3_EQ(Float::sort(), named_float.get(std::vector<ZE>{ZE_INDEX(0)}).simplify().get_sort());
  EXPECT_Z3_NE(Index::sort(), named_float.get(std::vector<ZE>{ZE_INDEX(0)}).simplify().get_sort());
}

TEST(UnitTensorTest, Named3D) {
  Tensor named_int("named", std::vector<ZE>{ZE_INDEX(3), ZE_INDEX(4), ZE_INDEX(5)}, Integer::sort(32));
  EXPECT_Z3_EQ(Integer::sort(32), named_int.get(std::vector<ZE>{ZE_INDEX(0), ZE_INDEX(0), ZE_INDEX(0)}).simplify().get_sort());
  EXPECT_Z3_EQ(Integer::sort(32), named_int.get(std::vector<ZE>{ZE_INDEX(1), ZE_INDEX(2), ZE_INDEX(3)}).simplify().get_sort());
  EXPECT_Z3_EQ(Integer::sort(32), named_int.get(std::vector<ZE>{ZE_INDEX(2), ZE_INDEX(2), ZE_INDEX(2)}).simplify().get_sort());

  EXPECT_Z3_NE(Float::sort(), named_int.get(std::vector<ZE>{ZE_INDEX(0), ZE_INDEX(1), ZE_INDEX(2)}).simplify().get_sort());
  EXPECT_Z3_NE(Float::sort(), named_int.get(std::vector<ZE>{ZE_INDEX(1), ZE_INDEX(1), ZE_INDEX(1)}).simplify().get_sort());
}

TEST(UnitTensorTest, RotateDimensions) {
  Tensor named_int = Tensor("named", std::vector<ZE>{ZE_INDEX(3), ZE_INDEX(4), ZE_INDEX(5)}, Integer::sort(32));
  Tensor rotated_int = named_int.rotateDimensions();
  EXPECT_Z3_EQ(Integer::sort(32), rotated_int.get(std::vector<ZE>{ZE_INDEX(0), ZE_INDEX(0), ZE_INDEX(0)}).simplify().get_sort());
  EXPECT_Z3_EQ(ZE_INDEX(5), (ZE)rotated_int.getDim(0));
  EXPECT_Z3_EQ(ZE_INDEX(3), (ZE)rotated_int.getDim(1));
  EXPECT_Z3_EQ(ZE_INDEX(4), (ZE)rotated_int.getDim(2));

  ZE elem = named_int.get(std::vector<ZE>{ZE_INDEX(1), ZE_INDEX(2), ZE_INDEX(3)}).simplify();
  ZE same_elem = rotated_int.get(std::vector<ZE>{ZE_INDEX(3), ZE_INDEX(1), ZE_INDEX(2)}).simplify();
  ZE diff_elem = rotated_int.get(std::vector<ZE>{ZE_INDEX(1), ZE_INDEX(2), ZE_INDEX(3)}).simplify();
  EXPECT_Z3_EQ(elem, same_elem);
  EXPECT_Z3_NE(elem, diff_elem);
}

TEST(UnitTensorTest, Reshape) {
  Tensor named_int = Tensor("named", std::vector<ZE>{ZE_INDEX(3), ZE_INDEX(4), ZE_INDEX(5)}, Integer::sort(32));

  Tensor reshaped_int_3d = named_int.reshape(std::vector<ZE>{ZE_INDEX(2), ZE_INDEX(5), ZE_INDEX(6)});
  ZE elem = named_int.get(std::vector<ZE>{ZE_INDEX(1), ZE_INDEX(2), ZE_INDEX(3)}).simplify(); // 1 * (4 * 5) + 2 * (5) + 3 = 33
  ZE same_elem = reshaped_int_3d.get(std::vector<ZE>{ZE_INDEX(1), ZE_INDEX(0), ZE_INDEX(3)}).simplify(); // 1 * (5 * 6) + 0 * (6) + 3 = 33
  ZE diff_elem = reshaped_int_3d.get(std::vector<ZE>{ZE_INDEX(1), ZE_INDEX(2), ZE_INDEX(3)}).simplify();
  EXPECT_Z3_EQ(elem, same_elem);
  EXPECT_Z3_NE(elem, diff_elem);

  Tensor reshaped_int_2d = named_int.reshape(std::vector<ZE>{ZE_INDEX(6), ZE_INDEX(10)});
  elem = named_int.get(std::vector<ZE>{ZE_INDEX(1), ZE_INDEX(2), ZE_INDEX(3)}).simplify(); // 1 * (4 * 5) + 2 * (5) + 3 = 33
  same_elem = reshaped_int_2d.get(std::vector<ZE>{ZE_INDEX(3), ZE_INDEX(3)}).simplify(); // 3 * (10) + 3 = 33
  diff_elem = reshaped_int_2d.get(std::vector<ZE>{ZE_INDEX(1), ZE_INDEX(2)}).simplify();
  EXPECT_Z3_EQ(elem, same_elem);
  EXPECT_Z3_NE(elem, diff_elem);

  EXPECT_ANY_THROW(named_int.reshape(std::vector<ZE>{ZE_INDEX(8), ZE_INDEX(7)})); // different size(dim)
}

TEST(UnitTensorTest, Transpose) {
  Tensor named_int = Tensor("named", std::vector<ZE>{ZE_INDEX(4), ZE_INDEX(5)}, Integer::sort(32));
  Tensor transposed_int = named_int.transpose();

  ZE elem = named_int.get(std::vector<ZE>{ZE_INDEX(1), ZE_INDEX(2)}).simplify();
  ZE same_elem = transposed_int.get(std::vector<ZE>{ZE_INDEX(2), ZE_INDEX(1)}).simplify();
  ZE diff_elem = transposed_int.get(std::vector<ZE>{ZE_INDEX(1), ZE_INDEX(2)}).simplify();
  EXPECT_Z3_EQ(elem, same_elem);
  EXPECT_Z3_NE(elem, diff_elem);

  elem = named_int.get(std::vector<ZE>{ZE_INDEX(3), ZE_INDEX(3)}).simplify();
  same_elem = transposed_int.get(std::vector<ZE>{ZE_INDEX(3), ZE_INDEX(3)}).simplify();
  EXPECT_Z3_EQ(elem, same_elem);

  named_int = Tensor("named", std::vector<ZE>{ZE_INDEX(3), ZE_INDEX(4), ZE_INDEX(5)}, Integer::sort(32));
  transposed_int = named_int.transpose();
  EXPECT_DEATH(named_int.transpose(), ""); // only 2D tensors supported
}

TEST(UnitTensorTest, Convolution) {
  // TODO
  EXPECT_FALSE(true);
}

TEST(UnitTensorTest, Matmul) {
  // TODO
  EXPECT_FALSE(true);
}

TEST(UnitTensorTest, Refines) {
  // TODO
  EXPECT_FALSE(true);
}

TEST(UnitTensorTest, Affine) {
  // TODO
  EXPECT_FALSE(true);
}

TEST(UnitTensorTest, MkLambda) {
  // TODO
  EXPECT_FALSE(true);
}
