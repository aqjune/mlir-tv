#include "abstractops.h"
#include "debug.h"
#include "encode.h"
#include "memory.h"
#include "opts.h"
#include "print.h"
#include "smt.h"
#include "state.h"
#include "utils.h"
#include "value.h"
#include "vcgen.h"
#include "analysis.h"

#include "magic_enum.hpp"
#include <chrono>
#include <fstream>
#include <functional>
#include <map>
#include <optional>
#include <sstream>
#include <variant>
#include <vector>
#include <queue>

using namespace smt;
using namespace std;

namespace {
class Defer {
private:
  function<void()> fn;
public:
  Defer(function<void()> &&fn): fn(fn) {}
  ~Defer() { fn(); }
};

class ValidationInput {
public:
  mlir::FuncOp src, tgt;
  string dumpSMTPath;

  TypeMap<size_t> numBlocksPerType;
  unsigned int f32NonConstsCount, f64NonConstsCount;
  set<llvm::APFloat> f32Consts, f64Consts;
  bool f32HasInfOrNaN, f64HasInfOrNaN;
  vector<mlir::memref::GlobalOp> globals;
  bool isFpAddAssociative;
  bool unrollIntSum; // sum(arr) := arr[0] + ... + arr[arr.len-1]
  bool useMultisetForFpSum;
};


llvm::cl::opt<bool> arg_fp_add_associative("associative",
  llvm::cl::desc("Assume that floating point add is associative "
                 "(experimental)"),
  llvm::cl::init(false),
  llvm::cl::cat(MlirTvCategory));

llvm::cl::opt<bool> arg_unroll_int_sum("unroll-int-sum",
  llvm::cl::desc("Fully unroll summation of integer arrays whose sizes are"
                 " known to be constant"),
  llvm::cl::init(false),
  llvm::cl::cat(MlirTvCategory));

llvm::cl::opt<unsigned> arg_unroll_fp_sum_bound("unroll-fp-sum-bound",
  llvm::cl::desc("If the summation of floating point is to be unrolled after "
                 "abstraction refinement, specify the max array size."),
  llvm::cl::init(10),
  llvm::cl::cat(MlirTvCategory));

llvm::cl::opt<bool> arg_multiset("multiset",
  llvm::cl::desc("Use multiset when encoding the associativity of the floating"
                 " point addition"),  llvm::cl::Hidden,
  llvm::cl::init(false),
  llvm::cl::cat(MlirTvCategory));

llvm::cl::opt<bool> use_concrete_fp_encoding("use-concrete-fp",
  llvm::cl::desc("Use concrete IEEE 754 floating point encoding."),
  llvm::cl::init(false), llvm::cl::Hidden,
  llvm::cl::cat(MlirTvCategory));

llvm::cl::opt<string> arg_dump_smt_to("dump-smt-to",
  llvm::cl::desc("Dump SMT queries to"), llvm::cl::value_desc("path"),
  llvm::cl::cat(MlirTvCategory));

llvm::cl::opt<bool> arg_smt_use_all_logic("smt-use-all-logic",
  llvm::cl::desc("Use ALL Logic for SMT"),
  llvm::cl::init(false),
  llvm::cl::cat(MlirTvCategory));

llvm::cl::opt<unsigned> fp_bits("fp-bits",
  llvm::cl::desc("The number of bits for the abstract representation of "
                 "non-constant float and double values."),
  llvm::cl::init(31), llvm::cl::value_desc("number"),
  llvm::cl::cat(MlirTvCategory));

llvm::cl::opt<unsigned int> num_memblocks("num-memory-blocks",
  llvm::cl::desc("Number of memory blocks per type required to validate"
                 " translation (set 0 to determine it via analysis)"),
  llvm::cl::init(0), llvm::cl::value_desc("number"),
  llvm::cl::cat(MlirTvCategory));

llvm::cl::opt<bool> memref_inputs_simple("memref-inputs-simple",
  llvm::cl::desc("Assume that MemRef arguments point to distinct memory"
                 " blocks and their offsets are zero."),
  llvm::cl::init(false),
  llvm::cl::cat(MlirTvCategory));

llvm::cl::opt<unsigned int> max_unknown_dimsize("max-unknown-dimsize",
  llvm::cl::desc("Maximum dimension size for unknown shaped dimension"
                    "(default value: 50)"),
  llvm::cl::init(50), llvm::cl::value_desc("number"),
  llvm::cl::cat(MlirTvCategory));

llvm::cl::opt<unsigned int> max_tensor_size("max-tensor-size",
  llvm::cl::desc("Specify the maximum number of elements of a dynamically"
      " sized tensor tensor."),
  llvm::cl::init(10000),
  llvm::cl::cat(MlirTvCategory));

llvm::cl::opt<int> max_const_tensor_size("max-const-tensor-size",
  llvm::cl::desc("Specify the maximum number of elements of a constant tensor"
      " that mlir-tv is going to encode precisely."
      "Any non-splat constant tensor having more elements than this will be"
      " encoded as a fully unknown array, possibly introducing validation"
      " failures."
      " If set to -1, there is no such limit."),
  llvm::cl::init(-1),
  llvm::cl::cat(MlirTvCategory));

llvm::cl::opt<bool> be_succinct("succinct",
  llvm::cl::desc("Do not print input programs and counter examples."),
  llvm::cl::init(false),
  llvm::cl::cat(MlirTvCategory));

llvm::cl::opt<bool> no_arith_properties("no-arith-properties",
  llvm::cl::desc("Encode addf, mulf, divf, expf without arithmetic properties."
      "(check only shape transformation)"),
  llvm::cl::init(false),
  llvm::cl::cat(MlirTvCategory));
};

llvm::cl::opt<string> arg_verify_fn_name("compare-fn-name",
  llvm::cl::desc("Specify the name of a function to verify."
      " If not set, verify every function."),
  llvm::cl::value_desc("name"),
  llvm::cl::cat(MlirTvCategory));

static optional<string> checkFunctionSignatures(
    mlir::FuncOp src, mlir::FuncOp tgt) {
  if (src.getNumArguments() != tgt.getNumArguments())
    return "The source and target program have different number of arguments.";

  unsigned n = src.getNumArguments();
  for (unsigned i = 0; i < n; ++i) {
    auto srcArgTy = src.getArgument(i).getType();
    auto tgtArgTy = tgt.getArgument(i).getType();
    if (srcArgTy != tgtArgTy) {
      string msg;
      TO_STRING(msg, "The source and target argument type is different.\n"
          << "Src: " << srcArgTy << ", Tgt: " << tgtArgTy);
      return msg;
    }
  }

  return {};
}

static State createInputState(
    mlir::FuncOp fn, std::unique_ptr<Memory> &&initMem,
    ArgInfo &args, vector<Expr> &preconds) {
  State s(move(initMem));
  unsigned n = fn.getNumArguments();
  TypeMap<unsigned> numMemRefArgs;

  for (unsigned i = 0; i < n; ++i) {
    auto arg = fn.getArgument(i);
    auto argty = arg.getType();

    if (auto value = args.get(i)) {
      // Use identical arguments from source when encoding target.
      if (holds_alternative<MemRef>(*value)) {
        MemRef memref = get<MemRef>(*value);
        memref.setMemory(s.m.get());
        s.regs.add(arg, move(memref));
      } else {
        s.regs.add(arg, move(*value));
      }
      continue;
    }

    // Encode arguments of the source function.
    if (auto ty = argty.dyn_cast<mlir::TensorType>()) {
      if (!Tensor::isTypeSupported(ty))
        throw UnsupportedException(ty);

      // Create fresh variables for unknown dimension sizes
      auto dims = ShapedValue::getDims(ty);
      auto tensor = Tensor::var(ty.getElementType(),
          "arg" + to_string(arg.getArgNumber()),
          dims);

      preconds.push_back(tensor.getWellDefined());
      s.regs.add(arg, move(tensor));

    } else if (auto ty = argty.dyn_cast<mlir::MemRefType>()) {
      if (!MemRef::isTypeSupported(ty))
        throw UnsupportedException(ty);

      // Create fresh variables for unknown dimension sizes
      auto dims = ShapedValue::getDims(ty);
      auto layout = MemRef::getLayout(ty, dims);

      // TODO : out of bounds pointer is allowed?
      auto memref = MemRef(s.m.get(), ty.getElementType(),
          "arg" + to_string(arg.getArgNumber()), dims, layout);

      if (memref_inputs_simple) {
        s.addPrecondition(((Expr)memref.getOffset()).isZero());
        unsigned constBID = numMemRefArgs[ty.getElementType()]++;
        s.addPrecondition(memref.getBID() == constBID);
      }

      // Function argument MemRefs must point to global memblocks.
      preconds.push_back(memref.isGlobalBlock());
      preconds.push_back(memref.getWellDefined());
      s.regs.add(arg, move(memref));

    } else {
      if (convertPrimitiveTypeToSort(argty) == nullopt) {
        throw UnsupportedException(arg.getType());
      }

      string name = "arg" + to_string(arg.getArgNumber());
      auto varty = VarType::UNBOUND;
      if (auto ty = argty.dyn_cast<mlir::IndexType>()) {
        s.regs.add(arg, Index::var(move(name), varty));

      } else if (auto ty = argty.dyn_cast<mlir::FloatType>()) {
        s.regs.add(arg, Float::var(move(name), argty, varty));

      } else if (auto ty = argty.dyn_cast<mlir::IntegerType>()) {
        unsigned bw = ty.getIntOrFloatBitWidth();
        s.regs.add(arg, Integer::var(move(name), bw, varty));

      } else {
        llvm::errs() << "convertPrimitiveTypeToSort must have returned nullopt"
                        " for this type!";
        abort();
      }
    }
    args.add(i, s.regs.findOrCrash(arg));
  }
  return s;
}

static pair<CheckResult, int64_t> solve(
    Solver &solver, const Expr &refinement_negated,
    const string &dumpSMTPath, const string &dump_string_to_suffix) {
  //solver.reset();
  solver.add(refinement_negated);

  if (!dumpSMTPath.empty()) {
#if SOLVER_Z3
    if (refinement_negated.hasZ3Expr() && solver.z3) {
      ofstream fout(dumpSMTPath + ".z3." + dump_string_to_suffix + ".smt2");
      fout << solver.z3->to_smt2();
      fout.close();
    }
#endif
#if SOLVER_CVC5
    if (refinement_negated.hasCVC5Term()) {
      ofstream fout(dumpSMTPath + ".cvc5." + dump_string_to_suffix + ".smt2");
      fout << refinement_negated.getCVC5Term();
      fout.close();
    }
#endif
  }

  auto startTime = chrono::system_clock::now();
  CheckResult result = solver.check();
  auto elapsedMillisec =
      chrono::duration_cast<chrono::milliseconds>(
        chrono::system_clock::now() - startTime).count();

  return {result, elapsedMillisec};
}

static const char *SMT_LOGIC_QF  = "QF_AUFBV";
static const char *SMT_LOGIC     = "AUFBV";
static const char *SMT_LOGIC_ALL = "ALL";

static Results checkRefinement(
    const ValidationInput &vinput,
    const State &st_src, const State &st_tgt, Expr &&precond,
    bool useAllLogic, int64_t &elapsedMillisec) {
  mlir::FuncOp src = vinput.src;
  mlir::FuncOp tgt = vinput.tgt;
  auto fnname = src.getName().str();

  auto printErrorMsg = [&](Solver &s, CheckResult res, const char *msg,
                           vector<Expr> &&params, VerificationStep step,
                           unsigned retidx = -1,
                           optional<mlir::Type> memElemType = nullopt){
    if (res.isUnknown()) {
      llvm::outs() << "== Result: timeout ==\n";
    } else if (res.hasSat()) {
      llvm::outs() << "== Result: " << msg << "\n";

      if (!be_succinct.getValue()) {
        aop::evalConsts(s.getModel());
        printCounterEx(
            s.getModel(), params, src, tgt, st_src, st_tgt, step, retidx,
            memElemType);
      }
    } else {
      llvm_unreachable("unexpected result");
    }
  };

  useAllLogic |= st_src.hasConstArray || st_tgt.hasConstArray;
  const char *logic = useAllLogic ? SMT_LOGIC_ALL :
      ((st_src.hasQuantifier || st_tgt.hasQuantifier) ?
        SMT_LOGIC : SMT_LOGIC_QF);
  verbose("checkRefinement") << "use logic: " << logic << "\n";

  { // 1. Check UB
    verbose("checkRefinement") << "1. Check UB\n";
    Solver s(logic);
    auto not_refines =
        (st_src.isWellDefined() & !st_tgt.isWellDefined()).simplify();
    auto res = solve(s, precond & not_refines, vinput.dumpSMTPath,
                     fnname + ".1.ub");
    elapsedMillisec += res.second;
    if (res.first.isInconsistent()) {
      llvm::outs() << "== Result: inconsistent output!!"
                      " either MLIR-TV or SMT solver has a bug ==\n";
      return Results::INCONSISTENT;
    } else if (!res.first.hasUnsat()) {
      printErrorMsg(s, res.first, "Source is more defined than target", {},
                    VerificationStep::UB);
      return res.first.hasSat() ? Results::UB : Results::TIMEOUT;
    }
  }

  if (st_src.retValues.size() != 0) { // 2. Check the return values
    verbose("checkRefinement") << "2. Check return values\n";
    unsigned numret = st_src.retValues.size();
    assert(numret == st_tgt.retValues.size());
    for (unsigned i = 0; i < numret; ++i) {
      Solver s(logic);

      auto [refines, params] =
          ::refines(st_tgt.retValues[i], st_src.retValues[i]);

      auto not_refines =
        (st_src.isWellDefined() & st_tgt.isWellDefined() & !refines)
        .simplify();
      auto res = solve(s, precond & not_refines, vinput.dumpSMTPath,
                      fnname + ".2.retval." + to_string(i));
      elapsedMillisec += res.second;

      if (res.first.isInconsistent()) {
        llvm::outs() << "== Result: inconsistent output!!"
                        " either MLIR-TV or SMT solver has a bug ==\n";
        return Results::INCONSISTENT;
      } else if (!res.first.hasUnsat()) {
        string msg = "Return value mismatch";
        if (numret != 1)
          msg = msg + " (" + to_string(i + 1) + "/" + to_string(numret) + ")";

        printErrorMsg(s, res.first, msg.c_str(), move(params),
                      VerificationStep::RetValue, i);
        return res.first.hasSat() ? Results::RETVALUE : Results::TIMEOUT;
      }
    }
  }

  if (st_src.m->getTotalNumBlocks() > 0 ||
      st_tgt.m->getTotalNumBlocks() > 0) { // 3. Check memory refinement
    verbose("checkRefinement") << "3. Check memory refinement\n";
    Solver s(logic);
    auto refinementPerType = st_tgt.m->refines(*st_src.m);
    // [refines, params]
    for (auto &[elementType, refinement]: refinementPerType) {
      Expr refines = refinement.first;
      auto &params = refinement.second;

      auto not_refines =
        (st_src.isWellDefined() & st_tgt.isWellDefined() & !refines).simplify();
      auto res = solve(s, precond & not_refines, vinput.dumpSMTPath,
                      fnname + ".3.memory." + to_string(elementType));
      elapsedMillisec += res.second;
      if (res.first.isInconsistent()) {
        llvm::outs() << "== Result: inconsistent output!!"
                        " either MLIR-TV or SMT solver has a bug ==\n";
        return Results::INCONSISTENT;

      } else if (!res.first.hasUnsat()) {
        printErrorMsg(s, res.first, "Memory mismatch", move(params),
                      VerificationStep::Memory, -1, elementType);
        return res.first.hasSat() ? Results::RETVALUE : Results::TIMEOUT;
      }
    }
  }

  return Results::SUCCESS;
}

static void printUnsupported(const UnsupportedException &ue) {
  auto obj = ue.getObject();
  string reason = ue.getReason();

  if (holds_alternative<mlir::Operation*>(obj)) {
    mlir::Operation *op = get<0>(obj);

    if (op == nullptr) {
      llvm::errs() << "This function is not supported.\n";
    } else {
      llvm::errs() << "Unknown op (" << op->getName() << "): " << *op << "\n";
    }
    if (!reason.empty())
      llvm::errs() << "\t" << reason << "\n";

  } else {
    mlir::Type ty = get<1>(obj);
    llvm::errs() << "Unsupported type: " << ty << "\n";
    if (!reason.empty())
      llvm::errs() << "\t" << reason << "\n";
  }
}

static State encodeFinalState(
    const ValidationInput &vinput, unique_ptr<Memory> &&initMem,
    bool printOps, bool issrc, ArgInfo &args, vector<Expr> &preconds) {
  mlir::FuncOp fn = issrc ? vinput.src : vinput.tgt;

  State st = createInputState(fn, move(initMem), args, preconds);

  if (printOps)
    llvm::outs() << (issrc ? "<src>" : "<tgt>") << "\n";

  encode(st, fn, printOps);

  return st;
}

// 'conjunction' overlaps with std::conjunction
// Will move this function to Expr::and someday
Expr exprAnd(const vector<Expr>& v) {
  Expr e = Expr::mkBool(true);
  for (auto &e2: v)
    e = e2 & e;
  return e;
}

static tuple<State, State, Expr> encodeFinalStates(
    const ValidationInput &vinput, bool printOps) {
  auto src = vinput.src, tgt = vinput.tgt;

  if (auto errmsg = checkFunctionSignatures(src, tgt))
    throw UnsupportedException(move(*errmsg));

  ArgInfo args;
  vector<Expr> preconds;

  // Set max. local blocks as num global blocks
  auto initMemSrc = make_unique<Memory>(
      vinput.numBlocksPerType, vinput.numBlocksPerType, vinput.globals);
  // Due to how CVC5 treats unbound vars, the initial memory must be precisely
  // copied
  unique_ptr<Memory> initMemTgt(initMemSrc->clone());
  initMemTgt->setIsSrc(false);

  State st_src = encodeFinalState(
      vinput, move(initMemSrc), printOps, true,  args, preconds);
  State st_tgt = encodeFinalState(
      vinput, move(initMemTgt), printOps, false, args, preconds);

  preconds.push_back(aop::getFpConstantPrecondition());

  if (aop::getFpAddAssociativity())
    preconds.push_back(aop::getFpAssociativePrecondition());

  if (aop::getFpCastIsPrecise())
    preconds.push_back(aop::getFpTruncatePrecondition());

  Expr precond =
      exprAnd(preconds) & st_src.precondition() & st_tgt.precondition();
  precond = precond.simplify();

  return {move(st_src), move(st_tgt), move(precond)};
}

static Results tryValidation(
    const ValidationInput &vinput, bool printOps, bool useAllLogic,
    int64_t &elapsedMillisec) {
  auto enc = encodeFinalStates(vinput, printOps);
  return checkRefinement(
        vinput, get<0>(enc), get<1>(enc), move(get<2>(enc)), useAllLogic,
        elapsedMillisec);
}

static void checkIsSrcAlwaysUB(
    const ValidationInput &vinput, bool wasSuccess, bool useAllLogic,
    int64_t &elapsedMillisec) {
  mlir::FuncOp src = vinput.src;
  string fnname = src.getName().str();

  // Set the abstract level to be as concrete as possible because we may not
  // be able to detect always-UB cases
  aop::Abstraction concreteAbs = {
    .fpDot = aop::AbsLevelFpDot::SUM_MUL,
    .fpCast = aop::AbsLevelFpCast::PRECISE,
    .intDot = aop::AbsLevelIntDot::SUM_MUL,
    .fpAddSumEncoding =
        vinput.isFpAddAssociative ?
          aop::AbsFpAddSumEncoding::USE_SUM_ONLY:
          aop::AbsFpAddSumEncoding::UNROLL_TO_ADD
  };
  aop::setAbstraction(concreteAbs,
      vinput.isFpAddAssociative,
      vinput.unrollIntSum,
      no_arith_properties.getValue(),
      use_concrete_fp_encoding.getValue(),
      arg_unroll_fp_sum_bound.getValue(),
      vinput.f32NonConstsCount, vinput.f32Consts, vinput.f32HasInfOrNaN,
      vinput.f64NonConstsCount, vinput.f64Consts, vinput.f64HasInfOrNaN);
  aop::setEncodingOptions(vinput.useMultisetForFpSum);

  ArgInfo args_dummy;
  vector<Expr> preconds;
  // Set blocks as initially alive, since making them dead always makes the
  // program more undefined. (This may not be true if ptr-to-int casts exist,
  // but we don't have a plan to support that)
  auto initMemory = make_unique<Memory>(
      vinput.numBlocksPerType, vinput.numBlocksPerType, vinput.globals,
      /*blocks initially alive*/true);
  auto st = encodeFinalState(vinput, move(initMemory), false, true,
      args_dummy, preconds);

  useAllLogic |= st.hasConstArray;
  auto logic = useAllLogic ? SMT_LOGIC_ALL :
      (st.hasQuantifier ? SMT_LOGIC : SMT_LOGIC_QF);
  verbose("checkIsSrcAlwaysUB") << "use logic: " << logic << "\n";

  Solver s(logic);
  auto not_ub = st.isWellDefined().simplify();
  auto smtres = solve(s, exprAnd(preconds) & not_ub, vinput.dumpSMTPath,
                      fnname + ".notub");
  elapsedMillisec += smtres.second;

  if (smtres.first.isInconsistent()) {
    llvm::outs() << "== Result: inconsistent output!!"
                    " either MLIR-TV or SMT solver has a bug ==\n";
  } else if (smtres.first.hasUnsat()) {
    llvm::outs() << "== Result: correct (source is always undefined) ==\n";
  } else if (wasSuccess) {
    llvm::outs() << "== Result: correct ==\n";
  }
}

static Results validate(ValidationInput vinput) {
  llvm::outs() << "=========== Function "
      << vinput.src.getName() << " ===========\n\n";
  if (vinput.src.getNumArguments() != vinput.tgt.getNumArguments())
    throw UnsupportedException("source, target has different num arguments");

  int64_t elapsedMillisec = 0;
  Defer timePrinter([&]() {
    llvm::outs() << "solver's running time: " << elapsedMillisec
        << " msec.\n\n";
  });
  using namespace aop;
  auto printSematics = [](Abstraction &abs, Results &result) {
    verbose("validate")  << "** Verification Result: "
        << magic_enum::enum_name(result.code) << "\n";

    llvm::outs()
      << "\n--------------------------------------------------------------\n"
      << "  Abstractions used for the validation:\n"
      << "  - dot ops (fp): " << magic_enum::enum_name(abs.fpDot) << "\n"
      << "  - cast ops (fp): " << magic_enum::enum_name(abs.fpCast) << "\n"
      << "  - add/sum ops (fp): "
      << magic_enum::enum_name(abs.fpAddSumEncoding) << "\n"
      << "  - dot ops (int): " << magic_enum::enum_name(abs.intDot) << "\n"
      << "--------------------------------------------------------------\n\n";
  };

  Results result(Results::Code::TIMEOUT);
  // Use all logic when "smt-use-all-logic" or "use-concrete-fp" is on
  // since IEEE754 encoding with arrays need ALL logic.
  auto useAllLogic = arg_smt_use_all_logic.getValue()
      || use_concrete_fp_encoding.getValue();
  queue<Abstraction> queue;

  queue.push({AbsLevelFpDot::FULLY_ABS,
      AbsLevelFpCast::FULLY_ABS,
      AbsLevelIntDot::FULLY_ABS,
      vinput.isFpAddAssociative ? AbsFpAddSumEncoding::USE_SUM_ONLY :
                  AbsFpAddSumEncoding::DEFAULT});


  setEncodingOptions(vinput.useMultisetForFpSum);
  resetAbstractlyEncodedAttrs();

  unsigned itrCount = 0;
  const string dumpSMTPath = vinput.dumpSMTPath;

  while (!queue.empty()) {
    auto abs = queue.front();
    queue.pop();

    if (itrCount > 0)
      llvm::outs() << "Validating the transformation with a refined "
          "abstraction...\n";

    setAbstraction(abs,
        vinput.isFpAddAssociative,
        vinput.unrollIntSum,
        no_arith_properties.getValue(),
        use_concrete_fp_encoding.getValue(),
        arg_unroll_fp_sum_bound.getValue(),
        vinput.f32NonConstsCount, vinput.f32Consts, vinput.f32HasInfOrNaN,
        vinput.f64NonConstsCount, vinput.f64Consts, vinput.f64HasInfOrNaN);

    if (!dumpSMTPath.empty()) {
      vinput.dumpSMTPath = dumpSMTPath;
      if (itrCount > 0)
        vinput.dumpSMTPath += "_refined_" + to_string(itrCount);
    }

    bool printOps = itrCount == 0 && !be_succinct.getValue();
    auto res = tryValidation(vinput, printOps, useAllLogic, elapsedMillisec);
    printSematics(abs, res);
    if (res.code == Results::INCONSISTENT) {
      return res;
    } else if (res.code == Results::SUCCESS) {
      checkIsSrcAlwaysUB(vinput, res.code == Results::SUCCESS,
          useAllLogic, elapsedMillisec);
      return res;
    } else {
      result = res;
    }

    // Do refinement of the abstraction.
    // Perform refinement in a lock-step manner to avoid combinatorial explosion
    auto usedOps = aop::getUsedAbstractOps();
    auto nextAbs = abs;
    bool isChanged = false;
    /* 1. fp dot abstraction level */
    if (abs.fpDot == AbsLevelFpDot::FULLY_ABS) {
      if (usedOps.fpDot || vinput.isFpAddAssociative) {
        nextAbs.fpDot = AbsLevelFpDot::SUM_MUL;
        isChanged = true;
      }
    }
    /* 2. fp cast abstraction level */
    if (abs.fpCast == AbsLevelFpCast::FULLY_ABS) {
      if (usedOps.fpCastRound) {
        nextAbs.fpCast = AbsLevelFpCast::PRECISE;
        isChanged = true;
      }
    }
    /* 3. int dot abstraction level */
    if (abs.intDot == AbsLevelIntDot::FULLY_ABS) {
      if (usedOps.intDot && usedOps.intSum) {
        nextAbs.intDot = AbsLevelIntDot::SUM_MUL;
        isChanged = true;
      }
    }

    if (isChanged) {
      queue.push(nextAbs);
    } else {
      /* 4. fp add, sum encoding level */
      // Since UNROLL_TO_ADD may cause big slowdown, turn in off at the end
      // only.
      // Do not refine fpAddSumEncoding if it is USE_SUM_ONLY.
      //  USE_SUM_ONLY is used only when --associative is given, and it cannot
      //  be refined further.
      if (abs.fpAddSumEncoding == AbsFpAddSumEncoding::DEFAULT) {
        if (usedOps.fpSum) {
          nextAbs.fpAddSumEncoding = AbsFpAddSumEncoding::UNROLL_TO_ADD;
          queue.push(nextAbs);
        }
      }
    }

    ++itrCount;
  }

  if (result.code == Results::TIMEOUT)
    checkIsSrcAlwaysUB(vinput, false, useAllLogic, elapsedMillisec);

  // If verification failed even when with the most concrete semantics,
  // return the last result
  return result;
}

static vector<mlir::memref::GlobalOp> mergeGlobals(
    const map<string, mlir::memref::GlobalOp> &srcGlobals,
    const map<string, mlir::memref::GlobalOp> &tgtGlobals) {

  vector<mlir::memref::GlobalOp> mergedGlbs;

  for (auto &[name, glbSrc0]: srcGlobals) {
    auto glbSrc = glbSrc0; // Remove constness
    auto tgtItr = tgtGlobals.find(name);
    if (tgtItr == tgtGlobals.end()) {
      mergedGlbs.push_back(glbSrc);
      continue;
    }

    auto glbTgt = tgtItr->second;
    if (glbSrc.type() != glbTgt.type() ||
        glbSrc.isPrivate() != glbTgt.isPrivate() ||
        glbSrc.constant() != glbTgt.constant() ||
        glbSrc.initial_value() != glbTgt.initial_value()) {
      throw UnsupportedException(
          name + " has different signatures in src and tgt");
    }

    assert(glbSrc.type().hasStaticShape() &&
           "Global var must be statically shaped");

    mergedGlbs.push_back(glbSrc);
  }

  for (auto &[name, glbTgt0]: tgtGlobals) {
    auto glbTgt = glbTgt0;
    auto tgtItr = srcGlobals.find(name);
    if (tgtItr == srcGlobals.end()) {
      if (glbTgt.constant()) {
        mergedGlbs.push_back(glbTgt);
      } else
        throw UnsupportedException("Introducing new non-const globals "
            "is not supported");
    }
  }

  return mergedGlbs;
}

Results validate(
    mlir::OwningModuleRef &src, mlir::OwningModuleRef &tgt) {
  map<llvm::StringRef, mlir::FuncOp> srcfns, tgtfns;
  auto fillFns = [](map<llvm::StringRef, mlir::FuncOp> &m, mlir::Operation &op) {
    auto fnop = mlir::dyn_cast<mlir::FuncOp>(op);
    if (fnop && !fnop.isDeclaration()) {
      m[fnop.getName()] = fnop;
    }
  };
  llvm::for_each(*src, [&](auto &op) { fillFns(srcfns, op); });
  llvm::for_each(*tgt, [&](auto &op) { fillFns(tgtfns, op); });

  Results verificationResult = Results::SUCCESS;
  bool hasUnsupported = false;

  llvm::StringRef verify_fn_name = llvm::StringRef(arg_verify_fn_name.getValue());
  bool is_check_single_fn = !verify_fn_name.empty();

  for (auto [name, srcfn]: srcfns) {
    if (is_check_single_fn) {
      if (!name.equals(verify_fn_name)) {
        continue;
      }
    }

    auto itr = tgtfns.find(name);
    if (itr == tgtfns.end()) {
      // The function does not exist in tgt! Let's skip this.
      // TODO: we should notify users that the functions are not checked.
      continue;
    }
    // TODO: check fn signature
    auto tgtfn = itr->second;

    AnalysisResult src_res, tgt_res;
    vector<mlir::memref::GlobalOp> globals;

    Tensor::MAX_TENSOR_SIZE = max_tensor_size.getValue();
    Tensor::MAX_CONST_SIZE = max_const_tensor_size.getValue();
    Tensor::MAX_DIM_SIZE = max_unknown_dimsize.getValue();
    MemRef::MAX_DIM_SIZE = max_unknown_dimsize.getValue();

    try {
      src_res = analyze(srcfn);
      tgt_res = analyze(tgtfn);
      globals = mergeGlobals(
          src_res.memref.usedGlobals, tgt_res.memref.usedGlobals);
    } catch (UnsupportedException ue) {
      printUnsupported(ue);
      hasUnsupported = true;
      continue;
    }

    auto f32_consts = src_res.F32.constSet;
    f32_consts.merge(tgt_res.F32.constSet);
    auto f64_consts = src_res.F64.constSet;
    f64_consts.merge(tgt_res.F64.constSet);

    ValidationInput vinput;
    vinput.src = srcfn;
    vinput.tgt = tgtfn;
    vinput.dumpSMTPath = arg_dump_smt_to.getValue();
    vinput.globals = globals;

    vinput.numBlocksPerType = src_res.memref.argCount;
    for (auto &[ty, cnt]: src_res.memref.varCount)
      vinput.numBlocksPerType[ty] += cnt;
    for (auto &[ty, cnt]: tgt_res.memref.varCount)
      vinput.numBlocksPerType[ty] += cnt;

    if (vinput.numBlocksPerType.size() > 1) {
      llvm::outs() << "NOTE: mlir-tv assumes that memrefs of different element "
          "types do not alias. This can cause missing bugs.\n";
    }

    if (num_memblocks.getValue() != 0) {
      for (auto &[ty, cnt]: vinput.numBlocksPerType)
        cnt = num_memblocks.getValue();
    }

    if (fp_bits.getValue() != 0) {
      assert(fp_bits.getValue() < 32 && "Given fp bits are too large");
      vinput.f32NonConstsCount = vinput.f64NonConstsCount =
          1u << fp_bits.getValue();
    } else {
      // Count non-constant floating points whose absolute values are distinct.
      auto countNonConstFps = [](const auto& src_res, const auto& tgt_res,
          bool elemwise) {
        if (elemwise) {
          return src_res.argCount + // # of variables in argument lists
            src_res.varCount + tgt_res.varCount; // # of variables in registers
        } else {
          return src_res.argCount + // # of variables in argument lists
            src_res.varCount + tgt_res.varCount + // # of variables in registers
            src_res.elemsCount + tgt_res.elemsCount;
                // # of ShapedType elements count
        }
      };

      auto isElementwise = src_res.isElementwiseFPOps ||
                           tgt_res.isElementwiseFPOps;
      vinput.f32NonConstsCount =
          countNonConstFps(src_res.F32, tgt_res.F32, isElementwise);
      vinput.f64NonConstsCount =
          countNonConstFps(src_res.F64, tgt_res.F64, isElementwise);
    }
    vinput.f32Consts = f32_consts;
    vinput.f32HasInfOrNaN = src_res.F32.hasInfOrNaN | tgt_res.F32.hasInfOrNaN;
    vinput.f64Consts = f64_consts;
    vinput.f64HasInfOrNaN = src_res.F64.hasInfOrNaN | tgt_res.F64.hasInfOrNaN;
    vinput.isFpAddAssociative = arg_fp_add_associative.getValue();
    vinput.unrollIntSum = arg_unroll_int_sum.getValue();
    vinput.useMultisetForFpSum = arg_multiset.getValue();

    try {
      verificationResult.merge(validate(vinput));
    } catch (UnsupportedException ue) {
      printUnsupported(ue);
      hasUnsupported = true;
    }
  }

  if (hasUnsupported) {
    exit(UNSUPPORTED_EXIT_CODE);
  }

  return verificationResult;
}
