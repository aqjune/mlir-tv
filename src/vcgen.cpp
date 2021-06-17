#include "tensor.h"
#include "smt.h"
#include "state.h"
#include "vcgen.h"

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/Matchers.h"
#include "z3++.h"
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <variant>
#include <vector>

using namespace std;


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
    auto argty = arg.getType();
    if (auto ty = argty.dyn_cast<mlir::TensorType>()) {
      s.regs.add(arg, Tensor("arg" + to_string(arg.getArgNumber()),
                             Tensor::getDims(ty)));
    } else if (auto ty = argty.dyn_cast<mlir::IndexType>()) {
      s.regs.add(arg, Index("arg" + to_string(arg.getArgNumber())));
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

  auto inputs = op.getInputTensorOperands();
  assert(inputs.size() == 2);
  auto input = inputs[0]->get();
  auto filter = inputs[1]->get();

  if (op.getOutputTensorOperands().size() != 1)
    return "operation with one output tensor is supported only";
  auto output = op.getOutputTensorOperands()[0]->get();

  auto t_input = st.regs.get<Tensor>(input);
  auto t_filter = st.regs.get<Tensor>(filter);

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
  st.regs.add(res, Tensor(name, Tensor::getDims(ty)));

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


  Tensor t_input = st.regs.get<Tensor>(op.getInputOperand(0)->get());
  vector<z3::expr> output_dimvars;
  vector<z3::expr> affine_exprs;

  for (unsigned i = 0; i < inputMap.getNumInputs(); ++i)
    output_dimvars.emplace_back(Index("i" + to_string(i)));
  for (unsigned i = 0; i < inputMap.getNumResults(); ++i) {
    auto ae_res = encodeAffineExpr(inputMap.getResult(i), output_dimvars);
    if (!ae_res)
      RET_STR_WITH_PREFIX("unsupported affine expr", inputMap.getResult(i));

    affine_exprs.emplace_back(move(*ae_res));
  }

  auto tensor_sz = Tensor::getDims(
      op.getOutputOperand(0)->get().getType().cast<mlir::TensorType>());
  Tensor t_res = t_input.affine(output_dimvars, affine_exprs, move(tensor_sz));
  st.regs.add(op.getOutputOperand(0)->get(), Tensor(t_res));
  st.regs.add(op.getResult(0), move(t_res));
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::linalg::TensorCollapseShapeOp op) {
  // TODO: is tensor_collapse_shape with permutated indices valid?
  //   ex: %2 = linalg.tensor_collapse_shape %1 [[0, 2], [1, 3]]
  // Then, it isn't simply reinterpretation of the operand; it needs permutation
  // of elements.

  Tensor t = st.regs.get<Tensor>(op.getOperand());
  st.regs.add(op.getResult(), t.reshape(Tensor::getDims(op.getResultType())));
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::linalg::TensorExpandShapeOp op) {
  // TODO: is tensor_expand_shape with permutated indices valid?
  //   ex: %2 = linalg.tensor_expand_shape %1 [[0], [2], [1, 3]]
  // Then, it isn't simply reinterpretation of the operand; it needs permutation
  // of elements.

  Tensor t = st.regs.get<Tensor>(op.getOperand());
  st.regs.add(op.getResult(), t.reshape(Tensor::getDims(op.getResultType())));
  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::linalg::MatmulOp op) {
  if (!op.hasTensorSemantics())
    return "operation with tensor semantics is supported only";

  if (op.getNumInputs() != 2 || op.getNumOutputs() != 1)
    return "unsupported form";

  Tensor a = st.regs.get<Tensor>(op.getOperand(0));
  Tensor b = st.regs.get<Tensor>(op.getOperand(1));
  Tensor result = a.matmul(b);
  st.regs.add(op.getResult(0), Tensor(result));
  st.regs.add(op.getOutputOperand(0)->get(), move(result));

  return {};
}

template<>
optional<string> encodeOp(State &st, mlir::ReturnOp op) {
  st.retValue = st.regs.get<Tensor>(op.getOperand(0));
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
  if (!llvm::hasSingleElement(fn.getRegion()))
    return "Only a function with one block is supported";

  auto &block = fn.getRegion().front();
  for (auto &op: block) {
    llvm::outs() << "  " << op << "\n";
    ENCODE(op, mlir::ReturnOp);
    ENCODE(op, mlir::linalg::ConvInputNHWCFilterHWCFOp);
    ENCODE(op, mlir::linalg::GenericOp);
    ENCODE(op, mlir::linalg::InitTensorOp);
    ENCODE(op, mlir::linalg::MatmulOp);
    ENCODE(op, mlir::linalg::TensorCollapseShapeOp);
    ENCODE(op, mlir::linalg::TensorExpandShapeOp);

    RET_STR("Unknown op: " << op);
  }
  llvm::outs() << "\n";

  return {};
}


static void printCounterEx(
    z3::solver &solver, const vector<z3::expr> &params, mlir::FuncOp src,
    State &st_src, State &st_src_in, State &st_tgt, State &st_tgt_in) {
  auto m = solver.get_model();
  auto or_omit = [&](const ValueTy &val) -> string {
    ValueTy evaluatedVal;
    visit([&](auto &&v) { evaluatedVal = v.eval(m); }, val);

    string s;
    llvm::raw_string_ostream rso(s);
    visit([&](auto &&v) { rso << v; }, evaluatedVal);
    rso.flush();

    if (s.size() > 500)
      return "(omitted)";
    return s;
  };

  llvm::outs() << "<Inputs>\n";

  unsigned n = src.getNumArguments();
  for (unsigned i = 0; i < n; ++i) {
    auto argsrc = src.getArgument(i);
    llvm::outs() << "\targ" << argsrc.getArgNumber() << ": "
                 << or_omit(st_src_in.regs.get<Tensor>(argsrc)) << "\n";
  }

  llvm::outs() << "\n<Source's variables>\n";
  for (auto &[v, e]: st_src.regs) {
    if (st_src_in.regs.contains(v))
      continue;
    llvm::outs() << "\t'" << v << "'\n\t\tValue: " << or_omit(e) << "\n";
  }

  llvm::outs() << "\n<Target's variables>\n";
  for (auto &[v, e]: st_tgt.regs) {
    if (st_tgt_in.regs.contains(v))
      continue;
    llvm::outs() << "\t'" << v << "'\n\t\tValue: " << or_omit(e) << "\n";
  }

  llvm::outs()
      << "\n<Return values>\n"
      << "\tIndex: " << solver.get_model().eval(params[0]) << "\n"
      << "\tSrc: " << or_omit(st_src.retValue)
      << "\n"
      << "\tTgt: " << or_omit(st_tgt.retValue)
      << "\n";

#if FALSE
  llvm::outs() << solver.get_model().to_string() << "\n";
#endif
}


static void verifyFunction(
    mlir::FuncOp src, mlir::FuncOp tgt, const string &dump_smt_to) {
  llvm::outs() << "Function " << src.getName() << "\n\n";
  assert(src.getNumArguments() == tgt.getNumArguments());

  auto raiseUnsupported = [](const string &msg) {
    llvm::errs() << msg << "\n";
    exit(1);
  };

  auto st_src_or_err = createInputState(src);
  if (holds_alternative<string>(st_src_or_err))
    raiseUnsupported(get<string>(st_src_or_err));
  auto st_src = get<State>(st_src_or_err);
  auto st_src_in = st_src; // for printing counter ex.

  auto st_tgt_or_err = createInputState(tgt);
  if (holds_alternative<string>(st_tgt_or_err))
    raiseUnsupported(get<string>(st_tgt_or_err));
  auto st_tgt = get<State>(st_tgt_or_err);
  auto st_tgt_in = st_tgt; // for printing counter ex.

  llvm::outs() << "<src>\n";
  if (auto msg = encode(st_src, src))
    raiseUnsupported(*msg);

  llvm::outs() << "<tgt>\n";
  if (auto msg = encode(st_tgt, tgt))
    raiseUnsupported(*msg);

  // Invoke Z3
  auto solver = z3::solver(ctx, "QF_UFBV");
  auto [refines, params] = st_tgt.refines(st_src);
  refines = refines.simplify();

  solver.add(!refines);

  if (!dump_smt_to.empty()) {
    ofstream fout(dump_smt_to + "." + src.getName().str());
    fout << refines;
    fout.close();
  }

  auto result = solver.check();
  if (result == z3::unsat) {
    llvm::outs() << "== Result: correct ==\n";
  } else if (result == z3::unknown) {
    llvm::outs() << "== Result: timeout ==\n";
  } else if (result == z3::sat) {
    llvm::outs() << "== Result: return value mismatch ==\n";
    printCounterEx(solver, params, src, st_src, st_src_in, st_tgt, st_tgt_in);
  }
}

void verify(mlir::OwningModuleRef &src, mlir::OwningModuleRef &tgt,
            const string &dump_smt_to) {
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
    verifyFunction(srcfn, itr->second, dump_smt_to);
  }
}