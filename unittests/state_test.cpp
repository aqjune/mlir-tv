#include "gtest/gtest.h"
#include "src/state.h"
#include "z3_expects.h"

#include "mlir/Parser.h"

class UnitRegFileTest : public ::testing::Test {
private:
  std::string sourceIR = 
  R""(
func @add_mul_fusion(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %f0 = constant 2.0 : f32
  %f1 = constant 3.0 : f32
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  return %arg2 : tensor<?x?xf32>
}
)"";

  std::vector<mlir::FuncOp> parseIR(std::string IR, mlir::MLIRContext *ctx) {
    src = mlir::parseSourceString(sourceIR, ctx);
    std::vector<mlir::FuncOp> srcFns;
    llvm::for_each(*src, [&](auto &op) { srcFns.push_back(mlir::dyn_cast<mlir::FuncOp>(op)); });
    return srcFns;
  }

  mlir::MLIRContext ctx;
  mlir::OwningModuleRef src;

protected:
  void SetUp() override {
    mlir::DialectRegistry registry;
    registry.insert<mlir::StandardOpsDialect>();
    registry.insert<mlir::tensor::TensorDialect>();
    ctx.appendDialectRegistry(registry);
    
    std::vector<mlir::ConstantIndexOp> indexOps;
    std::vector<mlir::ConstantFloatOp> floatOps;
    auto srcFns = parseIR(sourceIR, &ctx);
    auto &srcBlk = srcFns[0].getRegion().front();
    for (auto &op : srcBlk) {
      if (auto indexOp = mlir::dyn_cast<mlir::ConstantIndexOp>(op)) {
        indexOps.push_back(indexOp);
      } else if (auto floatOp = mlir::dyn_cast<mlir::ConstantFloatOp>(op)) {
        floatOps.push_back(floatOp);
      }
    }

    indexOp0 = indexOps[0];
    indexOp1 = indexOps[1];
    floatOp0 = floatOps[0];
    floatOp1 = floatOps[1];
    
    r2.add(indexOp0, Index(indexOp0.getValue()));
    r2.add(floatOp0, Float(floatOp0.getValue()));
    
    r3.add(indexOp0, Index(indexOp0.getValue()));
    r3.add(indexOp1, Index(indexOp1.getValue()));
    r3.add(floatOp0, Float(floatOp0.getValue()));
    r3.add(floatOp1, Float(floatOp1.getValue()));
  }

  // void TearDown() override {}

  RegFile r1;
  RegFile r2;
  RegFile r3;

  mlir::ConstantIndexOp indexOp0, indexOp1;
  mlir::ConstantFloatOp floatOp0, floatOp1;
};

TEST_F(UnitRegFileTest, IsEmptyInitially) {
  EXPECT_EQ(r1.begin(), r1.end());
}

TEST_F(UnitRegFileTest, Contains) {
  EXPECT_TRUE(r2.contains(indexOp0));
  EXPECT_FALSE(r2.contains(indexOp1));
  EXPECT_TRUE(r2.contains(floatOp0));
  EXPECT_FALSE(r2.contains(floatOp1));

  EXPECT_TRUE(r3.contains(indexOp0));
  EXPECT_TRUE(r3.contains(floatOp0));
  EXPECT_TRUE(r3.contains(floatOp1));
  EXPECT_TRUE(r3.contains(indexOp1));
}

TEST_F(UnitRegFileTest, Get) {
  EXPECT_NO_THROW(r2.get<Index>(indexOp0));
  EXPECT_THROW(r2.get<Float>(indexOp0), std::bad_variant_access);
  EXPECT_DEATH(r2.get<Index>(indexOp1), "Cannot find key"); // llvm_unreachable
  EXPECT_THROW(r2.get<Index>(floatOp0), std::bad_variant_access);
  EXPECT_NO_THROW(r2.get<Float>(floatOp0));
}

TEST_F(UnitRegFileTest, Iterator) {
  bool hasIndexOp0 = false, hasIndexOp1 = false;
  bool hasFloatOp0 = false, hasFloatOp1 = false;
  
  for (auto itr = r2.begin(); itr != r2.end(); itr++) {
    if (itr->first == indexOp0) {
      hasIndexOp0 = true;
    }
    if (itr->first == floatOp0) {
      hasFloatOp0 = true;
    }
  }
  EXPECT_TRUE(hasIndexOp0 && hasFloatOp0);

  hasIndexOp0 = false;
  hasFloatOp0 = false;
  r2.add(indexOp1, Index(indexOp1.getValue()));
  for (auto itr = r2.begin(); itr != r2.end(); itr++) {
    if (itr->first == indexOp0) {
      hasIndexOp0 = true;
    }
    if (itr->first == indexOp1) {
      hasIndexOp1 = true;
    }
    if (itr->first == floatOp0) {
      hasFloatOp0 = true;
    }
  }
  EXPECT_TRUE(hasIndexOp0 && hasIndexOp1 && hasFloatOp0);
}

TEST_F(UnitRegFileTest, GetZ3Expr) {
  EXPECT_Z3_EQ(r2.getZ3Expr(indexOp0), (z3::expr)Index(indexOp0.getValue()));
  EXPECT_DEATH(r2.getZ3Expr(indexOp1), "Cannot find key"); // llvm_unreachable
  EXPECT_Z3_EQ(r2.getZ3Expr(floatOp0), (z3::expr)Float(floatOp0.getValue()));

  EXPECT_Z3_EQ(r3.getZ3Expr(indexOp0), (z3::expr)Index(indexOp0.getValue()));
  EXPECT_Z3_EQ(r3.getZ3Expr(indexOp1), (z3::expr)Index(indexOp1.getValue()));
  EXPECT_Z3_EQ(r3.getZ3Expr(floatOp0), (z3::expr)Float(floatOp0.getValue()));
  EXPECT_Z3_EQ(r3.getZ3Expr(floatOp1), (z3::expr)Float(floatOp1.getValue()));
}
