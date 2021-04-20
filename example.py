from z3 import *
from functools import reduce
import time

INTRODUCE_BUG = False
#INTRODUCE_BUG = "value"
#INTRODUCE_BUG = "shape"
BITS_INDEX = 32
BITS_FLOAT = 4


tmpidx = 0
def nextTmpId() -> str:
  global tmpidx
  tmpidx = tmpidx + 1
  return tmpidx

def toBitVecs(idxs_const, bw):
  l = []
  for val in idxs_const:
    if is_bv(val):
      l.append(simplify(val))
    else:
      l.append(BitVecVal(val, bw))
  return l

def toPyInt(bv):
  if isinstance(bv, int):
    return bv
  if is_bv_value(bv):
    return bv.as_signed_long()
  return None

def isConstInt(bv, i=None):
  if i is None:
    return toPyInt(bv) != None
  return toPyInt(bv) == i

def to1DIdx(idxs, sizes):
  assert(len(idxs) == len(sizes))
  idx = idxs[0]
  for i in range(1, len(sizes)):
    if not isConstInt(sizes[i], 1) and not isConstInt(idx, 0):
      idx = idx * sizes[i]

    if isConstInt(idxs[i], 0):
      continue
    elif isConstInt(idx, 0):
      idx = idxs[i]
    else:
      idx = idx + idxs[i]

  return idx

def from1DIdx(idx, sizes):
  idxs = []

  for i in range(len(sizes) - 1, -1, -1):
    if isConstInt(sizes[i], 1):
      idxs.append(BitVecVal(0, BITS_INDEX))
      continue

    a = URem(idx, sizes[i])
    b = UDiv(idx, sizes[i])
    if isConstInt(idx) and isConstInt(sizes[i]):
      a = simplify(a)
      b = simplify(a)

    idxs.append(a)
    idx = b

  idxs.reverse()
  return idxs

def get1DSize(sizes):
  szaccml = BitVecVal(1, BITS_INDEX)
  for i in range(0, len(sizes)):
    szaccml = szaccml * sizes[i]
  szaccml = simplify(szaccml)
  return szaccml

def fitsInSize(idxs, sizes):
  preconds = []
  for i in range(0, len(idxs)):
    preconds.append(ULT(idxs[i], sizes[i]))
  return And(preconds)

def simplifyList(exprlist):
  exprlist = list(exprlist)
  for i in range(0, len(exprlist)):
    exprlist[i] = simplify(exprlist[i])
  return exprlist

def ForAllInRanges(idxs, sizes, body):
  assert(len(idxs) == len(sizes))
  idxs = list(idxs)
  sizes = list(sizes)
  i = 0
  while i < len(idxs):
    if isConstInt(sizes[i], 1):
      body = substitute(body, (idxs[i], BitVecVal(0, BITS_INDEX)))
      del sizes[i]
      del idxs[i]
    else:
      i = i + 1

  return ForAll(idxs, Implies(fitsInSize(idxs, sizes), 
  body))


class MemRef:
  @staticmethod
  def newVar(name: str, ns):
    val = Array("%s_val" % name, BitVecSort(BITS_INDEX),
                                 BitVecSort(BITS_FLOAT))
    return MemRef(ns, val)

  @staticmethod
  def mkLambda(ns, indexvars, body):
    assert(len(ns) == len(indexvars))

    idx1dvar = BitVec("idx", BITS_INDEX)
    idxvars = from1DIdx(idx1dvar, ns)
    body = substitute(body, *zip(indexvars, idxvars))

    return MemRef(ns, Lambda([idx1dvar], body))

  def __init__(self, ns, val):
    self.ns = ns
    self.val = val

  def _dump(self, indices):
    if len(indices) == len(self.ns):
      print("value %s: %s" % (str(indices), simplify(self.get(indices))))
    else:
      l = len(indices)
      for i in range(0, toPyInt(self.ns[l])):
        indices.append(i)
        self._dump(indices)
        indices.pop()

  def dump(self):
    print("(dim: %s, val: %s)" % (str(self.ns), str(self.val)))
    if not all([isConstInt(i) for i in self.ns]):
      print("Cannot dump elements")
    else:
      self._dump([])

  def __str__(self):
    sval = str(self.val)
    if (len(sval) > 100):
      sval = "(omitted)"
    return "memref(dim: %s, val: %s)" % (str(self.ns), sval)

  def evaluateFromModel(self, model):
    return MemRef([model.evaluate(n) for n in self.ns],
                  model.evaluate(self.val))

  def get(self, idxs):
    ret = Select(self.val, to1DIdx(idxs, self.ns))
    return ret

  def to1DArrayWithOfs(self, idxs, sizes):
    if all([isConstInt(x) for x in self.ns]) and \
       all([isConstInt(x) for x in sizes]):
      sz1 = toPyInt(get1DSize(sizes))
      sz2 = toPyInt(get1DSize(self.ns))
      assert(sz1 <= sz2), \
            "new size cannot be larger than the original size: %s vs. %s" % \
              (str(sz1), str(sz2))

    assert(len(idxs) == len(sizes))

    idxvar = BitVec("idx", BITS_INDEX)
    indices = from1DIdx(idxvar, sizes)
    return Lambda([idxvar], If(ULT(idxvar, get1DSize(sizes)), self.get(indices), 0))

  def affine(self, newidxvars, newtgtidxs, newsizes):
    idxvar = BitVec("idx", BITS_INDEX)
    indices = from1DIdx(idxvar, newsizes)

    newtgtidxs = list(newtgtidxs)
    for i in range(0, len(newtgtidxs)):
      newv = newtgtidxs[i]
      for j in range(0, len(newidxvars)):
        newv = substitute(newv, (newidxvars[j], indices[j]))
      newtgtidxs[i] = newv

    newm = MemRef(newsizes, Lambda([idxvar],
        If(ULT(idxvar, get1DSize(newsizes)),
           self.get(newtgtidxs), 0)))
    return newm

  def reshape(self, ns2):
    # Supported only when self.ns is constant
    newm = MemRef(simplifyList(ns2), self.val)
    return newm


def dot(a, b, n):
  # a, b: Array
  bis = BitVecSort(BITS_INDEX)
  bfs = BitVecSort(BITS_FLOAT)
  ars = ArraySort(bis, bfs)
  i = BitVec("idx", bis)
  dotfn = Function("dot", ars, ars, bfs)
  return dotfn(
    Lambda([i], If(ULT(i, n), Select(a, i), 0)),
    Lambda([i], If(ULT(i, n), Select(b, i), 0)))


def rotate(mr: MemRef):
  output_ns = list(mr.ns)
  output_ns.insert(0, output_ns.pop())

  i, j, k, l = BitVecs("i j k l", BITS_INDEX) # h, w, c, f
  output = mr.affine([l, i, j, k], [i, j, k, l], output_ns)
  return output


def convolution(inp: MemRef, filtr: MemRef, shapechks):
  # TODO: expand this to an input with batch size > 1
  assert(len(inp.ns) == 4)
  assert(len(filtr.ns) == 4)
  assert(isConstInt(inp.ns[0], 1)), "Unknown value: %s" % inp.ns
  assert(not isConstInt(
    simplify(And(ULE(filtr.ns[0], inp.ns[1]), ULE(filtr.ns[1], inp.ns[2]),
                 filtr.ns[2] == inp.ns[3])), 0)), \
    "Size mismatch: image: %s, filter: %s" % (str(inp.ns), str(filtr.ns))

  shapechks.append((inp.ns[3] == filtr.ns[2], "convol1")) # the num of channels
  shapechks.append((ULE(filtr.ns[0], inp.ns[1]), "convol2")) # height
  shapechks.append((ULE(filtr.ns[1], inp.ns[2]), "convol3")) # width

  output_ns = toBitVecs([
      1,           # TODO: support an input with batch size > 1
      inp.ns[1] + 1 - filtr.ns[0],
      inp.ns[2] + 1 - filtr.ns[1],
      filtr.ns[3]], BITS_INDEX) # channel(inp.ns[3] = filtr.ns[2]) disappears
  cube_size = [1] + filtr.ns[0:3]

  i, j, k, l = BitVecs("i j k l", BITS_INDEX) # n, h, w, f
  input_cube = inp.to1DArrayWithOfs(
      [0, j, k, 0], # batch: 0, img size: (h, w), channel starts from zero
      cube_size)

  # get l'th filter
  filter_cube = rotate(filtr).to1DArrayWithOfs([l, 0, 0, 0], cube_size)

  res = dot(input_cube, filter_cube,
            cube_size[0] * cube_size[1] * cube_size[2] * cube_size[3])
  output = MemRef.mkLambda(output_ns, [i, j, k, l], res)

  return output


def convertImageToMatrix(inp: MemRef, filtr_ns, shapechks):
  shapechks.append((ULE(filtr_ns[0], inp.ns[1]), "img2mat1")) # height
  shapechks.append((ULE(filtr_ns[1], inp.ns[2]), "img2mat2")) # width

  newsize = [
    inp.ns[0],
    inp.ns[1] - filtr_ns[0] + 1,
    inp.ns[2] - filtr_ns[1] + 1,
    filtr_ns[0],
    filtr_ns[1],
    filtr_ns[2]
  ]
  newsize = [simplify(x) for x in newsize]
  i, j, k, l, m, n = BitVecs("i j k l m n", BITS_INDEX)
  mat = inp.affine([i, j, k, l, m, n], [i, j + l, k + m, n], newsize)

  return mat


def reshape(a: MemRef, newsize, shapechks):
  shapechks.append((get1DSize(a.ns) == get1DSize(newsize), "reshape"))
  return a.reshape(newsize)

def transpose(a: MemRef):
  assert(len(a.ns) == 2)
  ns = [a.ns[1], a.ns[0]]
  i, j = BitVecs("i j", BITS_INDEX)
  return MemRef.mkLambda(ns, [i, j], a.get([j, i]))

def matmul(a: MemRef, b: MemRef, shapechks):
  assert(len(a.ns) == 2 and len(b.ns) == 2)

  if INTRODUCE_BUG == "value":
    bt = reshape(b, [b.ns[1], b.ns[0]], shapechks)
  elif INTRODUCE_BUG == "shape":
    bt = b
  else:
    bt = transpose(b)

  shapechks.append((a.ns[1] == bt.ns[1], "matmul"))

  i, j = BitVecs("i j", BITS_INDEX)
  a_row = a.to1DArrayWithOfs([i, 0], [1, a.ns[1]])
  bt_row = bt.to1DArrayWithOfs([j, 0], [1, bt.ns[1]])
  return MemRef.mkLambda([a.ns[0], bt.ns[0]], [i, j],
      dot(a_row, bt_row, a.ns[1]))



# Inputs
testcase = 0
input_preconds = []

if testcase == 0:
  imagesz = [1, 16, 16, 4]
  filtrsz = [3, 3, 4, 16]
elif testcase == 1:
  # simplest
  imagesz = [1, 4, 4, 1]
  filtrsz = [3, 3, 1, 1]
elif testcase == 2:
  # channel is 2
  imagesz = [1, 4, 4, 2]
  filtrsz = [3, 3, 2, 1]
elif testcase == 3:
  # channel is 2, 2 filters
  imagesz = [1, 4, 4, 2]
  filtrsz = [3, 3, 2, 2]
elif testcase == 4:
  # larger image
  imagesz = [1, 6, 6, 2]
  filtrsz = [3, 3, 2, 2]
elif testcase == 5:
  # many filters
  imagesz = [1, 6, 6, 2]
  filtrsz = [3, 3, 2, 16]
elif testcase == 6:
  h=BitVec("h", BITS_INDEX)
  w=BitVec("w", BITS_INDEX)
  c=BitVec("c", BITS_INDEX)
  f=BitVec("f", BITS_INDEX)
  input_preconds = [ULE(h, 100), ULE(w, 100), ULE(c, 100), ULE(f, 100)]
  imagesz = [1, h, w, c]
  filtrsz = [5, 5, c, f]

s_input = dict()
s_input["image"] = MemRef.newVar("image", toBitVecs(imagesz, BITS_INDEX))
s_input["filtr"] = MemRef.newVar("filtr", toBitVecs(filtrsz, BITS_INDEX))

print("Input vars: ")
for i in s_input:
  print("\t%s: %s" % (i, s_input[i]))

s_src = dict(s_input)
s_tgt = dict(s_input)
shapechks_src = []
shapechks_tgt = []

# Source program
s_src["output"] = convolution(s_src["image"], s_src["filtr"], shapechks_src)

# Target program
s_tgt["mat"] = convertImageToMatrix(s_tgt["image"], s_tgt["filtr"].ns,
                                    shapechks_tgt)

image_val = s_tgt["image"]
filtr_val = s_tgt["filtr"]
s_tgt["mat2"] = reshape(s_tgt["mat"], [
    image_val.ns[0] * (image_val.ns[1] - filtr_val.ns[0] + 1)
                    * (image_val.ns[2] - filtr_val.ns[1] + 1),
    filtr_val.ns[0] * filtr_val.ns[1] * filtr_val.ns[2]
], shapechks_tgt)

s_tgt["filtr2"] = reshape(s_tgt["filtr"], [
    filtr_val.ns[0] * filtr_val.ns[1] * filtr_val.ns[2],
    filtr_val.ns[3]
], shapechks_tgt)

s_tgt["output"] = matmul(s_tgt["mat2"], s_tgt["filtr2"], shapechks_tgt)


# Make & prove the goal

s = SolverFor("QF_UFBV")
i_counterex = BitVec("i_counterex", BITS_INDEX)
timeStart = time.time()

def printFalseShapeChks(model):
  print("Cannot satisfy this condition(s) at target:")
  for chk in shapechks_tgt:
    if model.eval(chk[0]) == False:
      print("\t%s (at %s)" % (chk[0], chk[1]))

def printCounterExs(model):
  print("\n<Return values>")
  print("Src: %s" % (str(s_src["output"].evaluateFromModel(model))))
  print("Tgt: %s" % (str(s_tgt["output"].evaluateFromModel(model))))
  print("Location: %s" % str([
    simplify(model.eval(i)) for i in
      from1DIdx(i_counterex, s_src["output"].ns)]))

  print("\n<Source variables>")
  for v in s_src:
    if v in s_input or v == "output":
      continue
    print("%s: %s" % (v, str(s_src[v].evaluateFromModel(model))))

  print("\n<Target variables>")
  for v in s_tgt:
    if v in s_input or v == "output":
      continue
    print("%s: %s" % (v, str(s_tgt[v].evaluateFromModel(model))))


# Let's check shape mismatch first.

src_no_ub = And([i[0] for i in shapechks_src] + input_preconds)
tgt_no_ub = And([i[0] for i in shapechks_tgt])
neg_goal = And(src_no_ub, Not(tgt_no_ub))

s.push()
s.add(neg_goal)
result = s.check()

if result == sat:
  print("\n== Result: shape mismatch ==")
  printFalseShapeChks(s.model())

else:
  neg_goal = And(
    src_no_ub, # src has no undefined behavior
    Or( # output_src != output_tgt
      get1DSize(s_src["output"].ns) != get1DSize(s_tgt["output"].ns),
      And(ULT(i_counterex, get1DSize(s_src["output"].ns)),
          Select(s_src["output"].val, i_counterex) !=
          Select(s_tgt["output"].val, i_counterex))))

  s.pop()
  s.add(neg_goal)

  with open("dump.txt", mode='w') as f:
    f.write("\n" + str(neg_goal))
  with open("dump.smt2", mode='w') as f:
    f.write(s.to_smt2())

  result = s.check()

  print()
  if result == unsat:
    print("== Result: correct ==")
  elif result == unknown:
    print("== Result: Z3 gives up ==")
    print(s.reason_unknown())
  elif result == sat:
    print("== Result: return value mismatch ==")
    printCounterExs(s.model())

timeEnd = time.time()
print("\nRunning time: %s secs" % (timeEnd - timeStart))