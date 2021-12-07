#include "abstractops.h"
#include "debug.h"
#include "encode.h"
#include "memory.h"
#include "print.h"
#include "smt.h"
#include "state.h"
#include "utils.h"
#include "value.h"
#include "vcgen.h"
#include "analysis.h"

#include <chrono>
#include <fstream>
#include <functional>
#include <map>
#include <optional>
#include <sstream>
#include <variant>
#include <vector>

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
  vector<mlir::memref::GlobalOp> globals;
  bool isFpAddAssociative;
  bool unrollIntSum;
  bool useMultisetForFpSum;
};

};


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
      auto tensor = Tensor(ty.getElementType(),
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

static const char *SMT_LOGIC_QF  = "QF_UFBV";
static const char *SMT_LOGIC     = "UFBV";
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
      printCounterEx(
          s.getModel(), params, src, tgt, st_src, st_tgt, step, retidx,
          memElemType);
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
    unsigned numret = st_src.retValues.size();
    assert(numret == st_tgt.retValues.size());
    for (unsigned i = 0; i < numret; ++i) {
      Solver s(logic);

      optional<Expr> refines_opt;
      vector<Expr> params;
      visit([&](auto &&src, auto &&tgt) {
        auto typedSrc = (decltype(tgt)) src;
        tie(refines_opt, params) = tgt.refines(typedSrc);
      }, st_src.retValues[i], st_tgt.retValues[i]);

      Expr refines = move(*refines_opt);

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

static void raiseUnsupported(const UnsupportedException &ue) {
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

  exit(UNSUPPORTED_EXIT_CODE);
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

  if (aop::getFpAddAssociativity())
    preconds.push_back(aop::getFpAssociativePrecondition());

  if(aop::getUsedAbstractOps().fpUlt)
    preconds.push_back(aop::getFpUltPrecondition());

  Expr precond =
      exprAnd(preconds) & st_src.precondition() & st_tgt.precondition();
  precond = precond.simplify();

  return {move(st_src), move(st_tgt), move(precond)};
}

static Results tryValidation(
    const ValidationInput &vinput, bool printOps, bool useAllLogic,
    int64_t &elapsedMillisec) {
  auto enc = encodeFinalStates(vinput, true);
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
  aop::setAbstraction(
      aop::AbsLevelFpDot::SUM_MUL,
      aop::AbsLevelFpCast::PRECISE,
      aop::AbsLevelIntDot::SUM_MUL,
      aop::AbsLevelFpSum::ADD_ONLY,
      vinput.isFpAddAssociative,
      vinput.unrollIntSum,
      vinput.f32NonConstsCount, vinput.f32Consts,
      vinput.f64NonConstsCount, vinput.f64Consts);
  aop::setEncodingOptions(vinput.useMultisetForFpSum);

  ArgInfo args_dummy;
  vector<Expr> preconds;
  auto initMemory = make_unique<Memory>(
      vinput.numBlocksPerType, vinput.numBlocksPerType, vinput.globals);
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
  llvm::outs() << "Function " << vinput.src.getName() << "\n\n";
  assert(vinput.src.getNumArguments() == vinput.tgt.getNumArguments());

  int64_t elapsedMillisec = 0;
  Defer timePrinter([&]() {
    llvm::outs() << "solver's running time: " << elapsedMillisec << " msec.\n";
  });

  using namespace aop;
  // Don't enable fp add associativity even if vinput.associativeAdd is true
  // because simply encoding it as UF is more efficient.
  // We can turn it on in the next iteration.
  setAbstraction(
      AbsLevelFpDot::FULLY_ABS,
      AbsLevelFpCast::FULLY_ABS,
      AbsLevelIntDot::FULLY_ABS,
      AbsLevelFpSum::FULLY_ABS,
      /*isFpAddAssociative*/false,
      vinput.unrollIntSum,
      vinput.f32NonConstsCount, vinput.f32Consts,
      vinput.f64NonConstsCount, vinput.f64Consts);
  setEncodingOptions(/*useMultiset*/false);

  auto res = tryValidation(vinput, true, false, elapsedMillisec);

  if (res.code == Results::INCONSISTENT)
    return res;
  else if (res.code == Results::SUCCESS) {
    // Check whether it is always UB
    checkIsSrcAlwaysUB(vinput, res.code == Results::SUCCESS, false,
        elapsedMillisec);
    return res;
  }

  auto usedOps = aop::getUsedAbstractOps();
  bool fpAssocAdd = vinput.isFpAddAssociative;
  // dot = mul + sum?
  bool useSumMulForFpDot = usedOps.fpDot && usedOps.fpSum && usedOps.fpMul;
  bool useSumMulForIntDot = usedOps.intDot && usedOps.intSum; // Eh.. int mul?
  bool useAddFOnly = usedOps.fpSum;
  bool fpCastRound = usedOps.fpCastRound;
  bool tryRefinedAbstraction =
      fpAssocAdd || useSumMulForFpDot || useSumMulForIntDot || fpCastRound || useAddFOnly;

  if (!tryRefinedAbstraction)
    return res;

  // Refine the current abstraction.
  setAbstraction(
      (useSumMulForFpDot || fpAssocAdd) ?
          AbsLevelFpDot::SUM_MUL : AbsLevelFpDot::FULLY_ABS,
      fpCastRound ? AbsLevelFpCast::PRECISE : AbsLevelFpCast::FULLY_ABS,
      useSumMulForIntDot? AbsLevelIntDot::SUM_MUL: AbsLevelIntDot::FULLY_ABS,
      useAddFOnly ? AbsLevelFpSum::ADD_ONLY : AbsLevelFpSum::FULLY_ABS,
      fpAssocAdd,
      vinput.unrollIntSum,
      vinput.f32NonConstsCount, vinput.f32Consts,
      vinput.f64NonConstsCount, vinput.f64Consts);
  setEncodingOptions(vinput.useMultisetForFpSum);

  if (!vinput.dumpSMTPath.empty())
    vinput.dumpSMTPath += "_noabs";

  llvm::outs()
      << "\n===============================================================\n"
      << "  Giving more precise semantics to abstractly defined ops...\n"
      << "===============================================================\n\n";

  bool useAllLogic = fpAssocAdd;
  res = tryValidation(vinput, false, useAllLogic, elapsedMillisec);
  if (res.code == Results::SUCCESS || res.code == Results::TIMEOUT)
    // Check whether it is always UB
    checkIsSrcAlwaysUB(vinput, res.code == Results::SUCCESS, useAllLogic,
                       elapsedMillisec);
  return res;
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
    mlir::OwningModuleRef &src, mlir::OwningModuleRef &tgt,
    const string &dumpSMTPath,
    unsigned int numBlocksPerType,
    pair<unsigned, unsigned> fpBits, bool isFpAddAssociative,
    bool unrollIntSum,
    bool useMultiset) {
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
  for (auto [name, srcfn]: srcfns) {
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

    try {
      src_res = analyze(srcfn);
      tgt_res = analyze(tgtfn);
      globals = mergeGlobals(
          src_res.memref.usedGlobals, tgt_res.memref.usedGlobals);
    } catch (UnsupportedException ue) {
      raiseUnsupported(ue);
    }

    auto f32_consts = src_res.F32.constSet;
    f32_consts.merge(tgt_res.F32.constSet);
    auto f64_consts = src_res.F64.constSet;
    f64_consts.merge(tgt_res.F64.constSet);

    ValidationInput vinput;
    vinput.src = srcfn;
    vinput.tgt = tgtfn;
    vinput.dumpSMTPath = dumpSMTPath;
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

    if (numBlocksPerType) {
      for (auto &[ty, cnt]: vinput.numBlocksPerType)
        cnt = numBlocksPerType;
    }

    if (fpBits.first) {
      assert(fpBits.first < 32 && fpBits.second < 32 &&
             "Given fp bits are too large");
      vinput.f32NonConstsCount = 1u << fpBits.first;
      vinput.f64NonConstsCount = 1u << fpBits.second;
    } else {
      // Count non-constant floating points whose absolute values are distinct.
      auto countNonConstFps = [](const auto& src_res, const auto& tgt_res, const auto& ew) {
        if (ew) {
          return src_res.argCount + // # of variables in argument lists
            src_res.varCount + tgt_res.varCount; // # of variables in registers
        } else {
          return src_res.argCount + // # of variables in argument lists
            src_res.varCount + tgt_res.varCount + // # of variables in registers
            src_res.elemCounts + tgt_res.elemCounts; // # of ShapedType elements count
        }
      };

      auto isElementwise = src_res.isElementwiseFPOps || tgt_res.isElementwiseFPOps;
      vinput.f32NonConstsCount = countNonConstFps(src_res.F32, tgt_res.F32, isElementwise);
      vinput.f64NonConstsCount = countNonConstFps(src_res.F64, tgt_res.F64, isElementwise);
    }
    vinput.f32Consts = f32_consts;
    vinput.f64Consts = f64_consts;
    vinput.isFpAddAssociative = isFpAddAssociative;
    vinput.unrollIntSum = unrollIntSum;
    vinput.useMultisetForFpSum = useMultiset;

    try {
      verificationResult.merge(validate(vinput));
    } catch (UnsupportedException ue) {
      raiseUnsupported(ue);
    }
  }

  return verificationResult;
}
