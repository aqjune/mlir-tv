#include "tensor.h"
#include "vcgen.h"

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/Matchers.h"
#include "z3++.h"
#include <functional>
#include <map>
#include <sstream>
#include <variant>
#include <vector>

using namespace std;

struct RegFile {
  vector<pair<mlir::Value, Tensor>> m;

  void add(mlir::Value v, Tensor &&t) {
    m.emplace_back(v, move(t));
  }

  Tensor &get(mlir::Value v) {
    for (auto &[a, b]: m)
      if (a == v)
        return b;

    llvm::errs() << "Cannot find key: " << v << "\n";
    assert(false && "Unknown key");
  }
};

struct State {
  RegFile regs;

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream&, State &);
};

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, State &s) {
  for (auto itm: s.regs.m) {
    os << "Register: " << itm.first;
    os << "Value: " << itm.second << "\n";
  }
  return os;
}


#define RET_STR(V) { \
  string msg; \
  llvm::raw_string_ostream rso(msg); \
  rso << V; \
  rso.flush(); \
  return msg; \
}
#define RET_STR_WITH_PREFIX(PREFIX, V) { \
  string msg; \
  llvm::raw_string_ostream rso(msg); \
  rso << PREFIX << V; \
  rso.flush(); \
  return msg; \
}

static variant<string, State>
createInputState(mlir::FuncOp fn) {
  State s;
  unsigned n = fn.getNumArguments();
  for (unsigned i = 0; i < n; ++i) {
    auto arg = fn.getArgument(i);
    if (auto ty = arg.getType().dyn_cast<mlir::TensorType>()) {
      s.regs.add(arg, Tensor::newVar(ty, to_string(arg.getArgNumber())));
    } else {
      RET_STR("Unsupported type: " << arg.getType());
    }
  }

  return s;
}


optional<z3::expr> encodeAffineExpr(
    mlir::AffineExpr ae, const vector<z3::expr> &dimvars
) {
  switch (ae.getKind()) {
  case mlir::AffineExprKind::Add: {
    auto aboe = ae.dyn_cast<mlir::AffineBinaryOpExpr>();
    auto lhs = encodeAffineExpr(aboe.getLHS(), dimvars);
    auto rhs = encodeAffineExpr(aboe.getRHS(), dimvars);
    if (!lhs || !rhs)
      return {};
    return *lhs + *rhs;
  }
  case mlir::AffineExprKind::DimId: {
    auto ade = ae.dyn_cast<mlir::AffineDimExpr>();
    auto id = ade.getPosition();
    assert(id < dimvars.size());
    return dimvars[id];
  }
  default:
    // Unsupported
    return {};
  }
}

template<class T>
static optional<string> encodeOp(State &st, T op);

template<>
optional<string>
encodeOp(State &st, mlir::linalg::ConvInputNHWCFilterHWCFOp op) {
  if (!llvm::all_of(op.dilations(), [](auto i) { return i == 1; }))
    return "dilation isn't one\n";
  else if (!llvm::all_of(op.strides(), [](auto i) { return i == 1; }))
    return "strides isn't one\n";

  if (!op.hasTensorSemantics())
    return "tensor semantics is supported only";

  auto inputs = op.getInputTensors();
  if (inputs.size() != 2)
    return "operation with (input, filter) input tensors is supported only";
  auto input = inputs[0];
  auto filter = inputs[1];

  if (op.getNumOutputTensors() != 1)
    return "operation with one output tensor is supported only";
  auto output = op.getOutputTensors()[0];

  auto &t_input = st.regs.get(input);
  auto &t_filter = st.regs.get(filter);

  auto t_res = t_input.conv(t_filter);
  st.regs.add(op.getResult(0), move(t_res));
  // TODO: check whether this semantics is correct.
  st.regs.add(output, move(t_res));

  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::linalg::InitTensorOp op) {
  auto res = op.getResult();
  auto ty = res.getType().dyn_cast<mlir::TensorType>();
  assert(ty);

  // FIXME: can we use res's name?
  static int new_var_idx = 0;
  auto name = string("init_tensor_") + to_string(new_var_idx++);
  st.regs.add(res, Tensor::newVar(ty, name));

  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::linalg::GenericOp op) {
  if (!op.hasTensorSemantics())
    return "operation with tensor semantics is supported only";

  if (op.getNumInputs() != 1 || op.getNumOutputs() != 1)
    return "operation with single input and output is supported only";

  auto indexingMaps = op.indexing_maps().getValue();
  if (indexingMaps.size() != 2)
    // one for input, one for output
    return "unknown indexing map form";

  auto inputMap = indexingMaps[0].cast<mlir::AffineMapAttr>().getValue();
  auto outputMap = indexingMaps[1].cast<mlir::AffineMapAttr>().getValue();
  if (!outputMap.isIdentity())
    return "identity output map is supported only";

  // Match one block including 'yield' only
  // Referred linalg::RegionMatcher::matchAsScalarBinaryOp
  auto &region = op.region();
  if (!llvm::hasSingleElement(region))
    return "operation with one block is supported only";

  auto &block = region.front();
  if (block.getNumArguments() != 2 ||
      !block.getArgument(0).getType().isSignlessIntOrFloat() ||
      !block.getArgument(1).getType().isSignlessIntOrFloat())
    return "unsupported block arguments";

  auto &ops = block.getOperations();
  using mlir::m_Op;
  using mlir::matchers::m_Val;

  auto p = m_Op<mlir::linalg::YieldOp>(m_Val(block.getArgument(0)));
  if (!llvm::hasSingleElement(ops) || !p.match(&ops.back()))
    return "yield is allowed only";


  const Tensor &t_input = st.regs.get(op.getInput(0));
  vector<z3::expr> output_dimvars;
  vector<z3::expr> affine_exprs;

  for (unsigned i = 0; i < inputMap.getNumInputs(); ++i)
    output_dimvars.emplace_back(Tensor::newIdxVar("i" + to_string(i)));
  for (unsigned i = 0; i < inputMap.getNumResults(); ++i) {
    auto ae_res = encodeAffineExpr(inputMap.getResult(i), output_dimvars);
    if (!ae_res)
      RET_STR_WITH_PREFIX("unsupported affine expr", inputMap.getResult(i));

    affine_exprs.emplace_back(move(*ae_res));
  }

  auto tensor_sz = Tensor::getDims(
      op.getOutput(0).getType().cast<mlir::TensorType>());
  Tensor t_res = t_input.affine(output_dimvars, affine_exprs, move(tensor_sz));
  st.regs.add(op.getOutput(0), Tensor(t_res));
  st.regs.add(op.getResult(0), move(t_res));
  return {};
}


#define ENCODE(op, ty) { \
  if (auto op2 = mlir::dyn_cast<ty>(op)) { \
    auto errmsg = encodeOp(st, op2); \
    if (errmsg) { \
      RET_STR("Cannot encode " << op << "\n\t" << *errmsg << "\n"); \
    } \
    continue; \
  } \
}

static optional<string> encode(State &st, mlir::FuncOp &fn) {
  for (auto &region: fn) {
    for (auto &op: region) {
      op.dump();
      ENCODE(op, mlir::linalg::ConvInputNHWCFilterHWCFOp);
      ENCODE(op, mlir::linalg::InitTensorOp);
      ENCODE(op, mlir::linalg::GenericOp);
    }
  }
  return {};
}


static void verify(mlir::FuncOp src, mlir::FuncOp tgt) {
  llvm::outs() << "<" << src.getName() << ">\n";
  assert(src.getNumArguments() == tgt.getNumArguments());

  auto raiseUnsupported = [](const string &msg) {
    llvm::errs() << msg << "\n";
    exit(1);
  };

  auto st_src_or_err = createInputState(src);
  if (holds_alternative<string>(st_src_or_err))
    raiseUnsupported(get<string>(st_src_or_err));
  auto st_src = get<State>(st_src_or_err);

  auto st_tgt_or_err = createInputState(tgt);
  if (holds_alternative<string>(st_tgt_or_err))
    raiseUnsupported(get<string>(st_tgt_or_err));
  auto st_tgt = get<State>(st_tgt_or_err);

  llvm::outs() << st_src << "\n";

  llvm::outs() << "<src>\n";
  if (auto msg = encode(st_src, src))
    raiseUnsupported(*msg);

  llvm::outs() << "\n";
  llvm::outs() << "<tgt>\n";
  if (auto msg = encode(st_tgt, tgt))
    raiseUnsupported(*msg);

  // TODO: compare the final state
}

void verify(mlir::OwningModuleRef &src, mlir::OwningModuleRef &tgt) {
  map<llvm::StringRef, mlir::FuncOp> srcfns, tgtfns;
  auto fillFns = [](map<llvm::StringRef, mlir::FuncOp> &m, mlir::Operation &op) {
    auto fnop = mlir::dyn_cast<mlir::FuncOp>(op);
    m[fnop.getName()] = fnop;
  };
  llvm::for_each(*src, [&](auto &op) { fillFns(srcfns, op); });
  llvm::for_each(*tgt, [&](auto &op) { fillFns(tgtfns, op); });

  for (auto [name, srcfn]: srcfns) {
    auto itr = tgtfns.find(name);
    if (itr == tgtfns.end()) {
      // The function does not exist in tgt! Let's skip this.
      // TODO: we should notify users that the functions are not checked.
      continue;
    }
    // TODO: check fn signature
    verify(srcfn, itr->second);
  }
}