#include "abstractops.h"
#include "encode.h"
#include "print.h"
#include "smt.h"
#include "state.h"
#include "utils.h"
#include "value.h"
#include "vcgen.h"

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

#define RET_STR(V) { \
  string msg; \
  llvm::raw_string_ostream rso(msg); \
  rso << V; \
  rso.flush(); \
  return msg; \
}

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

  MemEncoding encoding;
  unsigned int numBlocks;
};

};


static optional<string> checkFunctionSignatures(
    mlir::FuncOp src, mlir::FuncOp tgt) {
  if (src.getNumArguments() != tgt.getNumArguments())
    RET_STR("The source and target program have different number of arguments.");

  unsigned n = src.getNumArguments();
  for (unsigned i = 0; i < n; ++i) {
    auto srcArgTy = src.getArgument(i).getType();
    auto tgtArgTy = tgt.getArgument(i).getType();
    if (srcArgTy != tgtArgTy)
      RET_STR("The source and target argument type is different.\n"
          << "Src: " << srcArgTy << ", Tgt: " << tgtArgTy);
  }

  return {};
}

static variant<string, State>
createInputState(
    mlir::FuncOp fn, unsigned int numBlocks, MemEncoding encoding,
    ArgInfo &args, vector<Expr> &preconds) {
  State s(numBlocks, encoding);
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

    // Encode each arguments of source.
    if (auto ty = argty.dyn_cast<mlir::TensorType>()) {
      auto dimsAndElemTy = Tensor::getDimsAndElemTy(ty);
      if (!dimsAndElemTy)
        RET_STR("Unsupported Tensor element type: " << arg.getType());
      auto tensor = Tensor("arg" + to_string(arg.getArgNumber()),
        dimsAndElemTy->first,
        dimsAndElemTy->second);
      preconds.push_back(tensor.getWellDefined());
      s.regs.add(arg, move(tensor));

    } else if (auto ty = argty.dyn_cast<mlir::MemRefType>()) {
      auto dimsAndLayoutAndElemTy = MemRef::getDimsAndLayoutAndElemTy(ty);
      if (!dimsAndLayoutAndElemTy)
        RET_STR("Unsupported MemRef element type: " << arg.getType());
      // TODO : out of bounds pointer is allowed?
      auto memref = MemRef(s.m.get(), "arg" + to_string(arg.getArgNumber()),
        get<0>(*dimsAndLayoutAndElemTy),
        get<1>(*dimsAndLayoutAndElemTy),
        get<2>(*dimsAndLayoutAndElemTy));
      // memref from function argument must point global memblock.
      preconds.push_back(memref.isGlobalBlock());
      preconds.push_back(memref.getWellDefined());
      s.regs.add(arg, move(memref));

    } else if (auto ty = argty.dyn_cast<mlir::IndexType>()) {
      s.regs.add(arg, Index("arg" + to_string(arg.getArgNumber())));

    } else if (auto ty = argty.dyn_cast<mlir::FloatType>()) {
      s.regs.add(arg, Float("arg" + to_string(arg.getArgNumber())));

    } else {
      RET_STR("Unsupported type: " << arg.getType());
    }
    args.add(i, s.regs.findOrCrash(arg));
  }
  return s;
}

static pair<CheckResult, int64_t> solve(
    Solver &solver, const Expr &refinement_negated,
    const string &dumpSMTPath, const string &dump_string_to_suffix) {
  solver.reset();
  solver.add(refinement_negated);

  if (!dumpSMTPath.empty()) {
    ofstream fout(dumpSMTPath + "." + dump_string_to_suffix);
    fout << refinement_negated;
    fout.close();
  }

  auto startTime = chrono::system_clock::now();
  CheckResult result = solver.check();
  auto elapsedMillisec =
      chrono::duration_cast<chrono::milliseconds>(
        chrono::system_clock::now() - startTime).count();

  return {result, elapsedMillisec};
}

static const char *SMT_LOGIC_QF = "QF_UFBV";
static const char *SMT_LOGIC    = "UFBV";

static Results checkRefinement(
    const ValidationInput &vinput,
    const State &st_src, const State &st_tgt, Expr &&precond,
    int64_t &elapsedMillisec) {
  mlir::FuncOp src = vinput.src;
  mlir::FuncOp tgt = vinput.tgt;
  auto fnname = src.getName().str();

  auto printErrorMsg = [&](Solver &s, CheckResult res, const char *msg,
                           vector<Expr> &&params, VerificationStep step,
                           unsigned retidx = -1){
    if (res.isUnknown()) {
      llvm::outs() << "== Result: timeout ==\n";
    } else if (res.isSat()) {
      llvm::outs() << "== Result: " << msg << "\n";
      printCounterEx(
          s.getModel(), params, src, tgt, st_src, st_tgt, step, retidx);
    } else {
      llvm_unreachable("unexpected result");
    }
  };
  const char *logic = (st_src.hasQuantifier || st_tgt.hasQuantifier) ?
      SMT_LOGIC : SMT_LOGIC_QF;

  { // 1. Check UB
    Solver s(logic);
    auto not_refines =
        (st_src.isWellDefined() & !st_tgt.isWellDefined()).simplify();
    auto res = solve(s, precond & not_refines, vinput.dumpSMTPath,
                     fnname + ".1.ub");
    elapsedMillisec += res.second;
    if (!res.first.isUnsat()) {
      printErrorMsg(s, res.first, "Source is more defined than target", {},
                    VerificationStep::UB);
      return res.first.isSat() ? Results::UB : Results::TIMEOUT;
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
        auto typedTarget = (decltype(src)) tgt;
        tie(refines_opt, params) = src.refines(typedTarget);
      }, st_src.retValues[i], st_tgt.retValues[i]);

      Expr refines = move(*refines_opt);

      auto not_refines =
        (st_src.isWellDefined() & st_tgt.isWellDefined() & !refines)
        .simplify();
      auto res = solve(s, precond & not_refines, vinput.dumpSMTPath,
                      fnname + ".2.retval." + to_string(i));
      elapsedMillisec += res.second;

      if (!res.first.isUnsat()) {
        string msg = "Return value mismatch";
        if (numret != 1)
          msg = msg + " (" + to_string(i + 1) + "/" + to_string(numret) + ")";

        printErrorMsg(s, res.first, msg.c_str(), move(params),
                      VerificationStep::RetValue, i);
        return res.first.isSat() ? Results::RETVALUE : Results::TIMEOUT;
      }
    }
  }

  if (st_src.m->getNumBlocks() > 0 ||
      st_tgt.m->getNumBlocks() > 0) { // 3. Check memory refinement
    Solver s(logic);
    auto [refines, params] = st_src.m->refines(*st_tgt.m);
    auto not_refines =
      (st_src.isWellDefined() & st_tgt.isWellDefined() & !refines).simplify();
    auto res = solve(s, precond & not_refines, vinput.dumpSMTPath,
                     fnname + ".3.memory");
    elapsedMillisec += res.second;
    if (!res.first.isUnsat()) {
      printErrorMsg(s, res.first, "Memory mismatch", move(params),
                    VerificationStep::Memory);
      return res.first.isSat() ? Results::RETVALUE : Results::TIMEOUT;
    }
  }

  return Results::SUCCESS;
}

static void raiseUnsupported(const string &msg) {
  llvm::errs() << msg << "\n";
  exit(91);
}

static State encodeFinalState(
    const ValidationInput &vinput, bool printOps, bool issrc, ArgInfo &args,
    vector<Expr> &preconds) {
  mlir::FuncOp fn = issrc ? vinput.src : vinput.tgt;

  auto st_or_err = createInputState(
      fn, vinput.numBlocks, vinput.encoding, args, preconds);
  if (holds_alternative<string>(st_or_err))
    raiseUnsupported(get<string>(st_or_err));
  auto st = get<State>(st_or_err);

  if (printOps)
    llvm::outs() << (issrc ? "<src>" : "<tgt>") << "\n";
  if (auto msg = encode(st, fn, printOps))
    raiseUnsupported(*msg);

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
    raiseUnsupported(*errmsg);

  ArgInfo args;
  vector<Expr> preconds;

  State st_src = encodeFinalState(vinput, printOps, true,  args, preconds);
  State st_tgt = encodeFinalState(vinput, printOps, false, args, preconds);

  Expr precond =
      exprAnd(preconds) & st_src.precondition() & st_tgt.precondition();
  precond = precond.simplify();

  return {move(st_src), move(st_tgt), move(precond)};
}

static Results tryValidation(
    const ValidationInput &vinput, bool printOps, int64_t &elapsedMillisec) {
  auto enc = encodeFinalStates(vinput, true);
  return checkRefinement(
        vinput, get<0>(enc), get<1>(enc), move(get<2>(enc)), elapsedMillisec);
}

static void checkIsSrcAlwaysUB(
    const ValidationInput &vinput, bool wasSuccess, int64_t &elapsedMillisec) {
  static bool isCalled = false;
  assert(!isCalled);
  isCalled = true;
  mlir::FuncOp src = vinput.src;
  string fnname = src.getName().str();

  // Set the abstract level to be as concrete as possible because we may not
  // be able to detect always-UB cases
  aop::setAbstractionLevel(aop::AbsLevelDot::SUM_MUL);

  ArgInfo args_dummy;
  vector<Expr> preconds;
  auto st = encodeFinalState(vinput, false, true, args_dummy, preconds);

  auto logic = st.hasQuantifier ? SMT_LOGIC : SMT_LOGIC_QF;
  Solver s(logic);
  auto not_ub = st.isWellDefined().simplify();
  auto smtres = solve(s, exprAnd(preconds) & not_ub, vinput.dumpSMTPath,
                      fnname + ".notub");
  elapsedMillisec += smtres.second;

  if (smtres.first.isUnsat()) {
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

  aop::setAbstractionLevel(aop::AbsLevelDot::FULLY_ABS);
  auto res = tryValidation(vinput, true, elapsedMillisec);

  if (res.code == Results::SUCCESS || res.code == Results::TIMEOUT) {
    // Check whether it is always UB
    checkIsSrcAlwaysUB(vinput, res.code == Results::SUCCESS, elapsedMillisec);
    return res;
  }

  auto usedOps = aop::getUsedAbstractOps();
  if (usedOps.dot && usedOps.sum && usedOps.mul) {
    // dot = mul + sum
    aop::setAbstractionLevel(aop::AbsLevelDot::SUM_MUL);
    if (!vinput.dumpSMTPath.empty())
      vinput.dumpSMTPath += "_noabs";
  } else
    return res;

  // Try more precise encoding
  llvm::outs()
      << "\n===============================================================\n"
      << "  Giving more precise semantics to abstractly defined ops...\n"
      << "===============================================================\n\n";

  res = tryValidation(vinput, false, elapsedMillisec);
  if (res.code == Results::SUCCESS || res.code == Results::TIMEOUT)
    // Check whether it is always UB
    checkIsSrcAlwaysUB(vinput, res.code == Results::SUCCESS, elapsedMillisec);
  return res;
}


Results validate(
    mlir::OwningModuleRef &src, mlir::OwningModuleRef &tgt,
    const string &dumpSMTPath, unsigned int numBlocks, MemEncoding encoding) {
  map<llvm::StringRef, mlir::FuncOp> srcfns, tgtfns;
  auto fillFns = [](map<llvm::StringRef, mlir::FuncOp> &m, mlir::Operation &op) {
    auto fnop = mlir::dyn_cast<mlir::FuncOp>(op);
    if (fnop.isDeclaration())
      return;
    m[fnop.getName()] = fnop;
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

    ValidationInput vinput;
    vinput.src = srcfn;
    vinput.tgt = itr->second;
    vinput.dumpSMTPath = dumpSMTPath;
    vinput.numBlocks = numBlocks;
    vinput.encoding = encoding;

    verificationResult.merge(validate(vinput));
  }

  return verificationResult;
}
