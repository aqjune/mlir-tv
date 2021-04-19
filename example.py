from z3 import *
import time

DEBUG=True
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
      l.append(val)
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

    idxs.append(URem(idx, sizes[i]))
    idx = UDiv(idx, sizes[i])

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
  def newVar(name: str, dims: int, **kwargs):
    ns = []
    if "ns" in kwargs:
      ns = simplifyList(kwargs["ns"])
      assert(len(ns) == dims)
    else:
      for i in range(0, dims):
        ns.append(BitVec("%s_dim%d" % (name, i), BITS_INDEX))
    val = Array("%s_val" % name, BitVecSort(BITS_INDEX),
                                 BitVecSort(BITS_FLOAT))
    return MemRef(ns, val)

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
    print("val: %s" % self.val)
    print("ns: %s" % self.ns)
    if not all([isConstInt(i) for i in self.ns]):
      print("Cannot dump values")
    else:
      self._dump([])

  def get(self, idxs):
    ret = Select(self.val, to1DIdx(idxs, self.ns))
    return ret

  def to1DArrayWithOfs(self, idxs, sizes):
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


def convolution(inp: MemRef, filtr: MemRef, preconds):
  # TODO: expand this to an input with batch size > 1
  assert(len(inp.ns) == 4)
  assert(len(filtr.ns) == 4)
  assert(isConstInt(inp.ns[0], 1)), "Unknown value: %s" % inp.ns

  output_ns = toBitVecs([
      1,           # TODO: support an input with batch size > 1
      inp.ns[1] + 1 - filtr.ns[0],
      inp.ns[2] + 1 - filtr.ns[1],
      filtr.ns[3]], BITS_INDEX) # channel(inp.ns[3] = filtr.ns[2]) disappears
  output = MemRef.newVar("conv_output%d" % nextTmpId(), 4, ns=output_ns)

  cube_size = [1] + filtr.ns[0:3]
  i, j, k, l = BitVecs("i j k l", BITS_INDEX) # h, w, c, f
  input_cube = inp.to1DArrayWithOfs(
      [0, i, j, 0], # batch: 0, img size: (h, w), channel starts from zero
      cube_size)

  # get l'th filter
  filter_cube = rotate(filtr).to1DArrayWithOfs([l, 0, 0, 0], cube_size)

  res = dot(input_cube, filter_cube,
            cube_size[0] * cube_size[1] * cube_size[2] * cube_size[3])
  idxs = [k, i, j, l]
  preconds.append(ForAllInRanges(idxs, output.ns, output.get(idxs) == res))

  if DEBUG:
    print("convolution(): result memref size: %s" % str(output.ns))

  return output


def convertImageToMatrix(inp: MemRef, filtr_ns, preconds):
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

  if DEBUG:
    print("convertImageToMatrix(): result memref size: %s" % str(mat.ns))

  return mat


def reshape(a: MemRef, newsize):
  return a.reshape(newsize)


def transpose(a: MemRef):
  assert(len(a.ns) == 2)
  ns = [a.ns[1], a.ns[0]]
  idx = BitVec("idx", BITS_INDEX)
  idxs = from1DIdx(idx, ns)
  return MemRef(ns, Lambda([idx], a.get([idxs[1], idxs[0]])))


def matmul(a: MemRef, b: MemRef, preconds):
  assert(len(a.ns) == 2 and len(b.ns) == 2)
  bt = transpose(b)

  output = MemRef.newVar("matmul%d" % nextTmpId(), 2, ns=[a.ns[0], bt.ns[0]])
  i, j = BitVecs("i j", BITS_INDEX)

  a_row = a.to1DArrayWithOfs([i, 0], [1, a.ns[1]])
  bt_row = bt.to1DArrayWithOfs([j, 0], [1, bt.ns[1]])

  preconds.append(
      ForAllInRanges([i, j], output.ns,
                     output.get([i, j]) == dot(a_row, bt_row, a.ns[1])))

  if DEBUG:
    print("matmul(): result memref size: %s" % str(output.ns))

  return output




# Inputs
testcase = 5
if testcase == 0:
  imagesz = [1, 16, 16, 4]
  filtrsz = [3, 3, 4, 16]
elif testcase == 1:
  # simplest
  imagesz = [1, 4, 4, 1]
  filtrsz = [3, 3, 2, 1]
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

s_src = dict()
s_src["image"] = MemRef.newVar("image",  4, ns=toBitVecs(imagesz, BITS_INDEX))
s_src["filtr"] = MemRef.newVar("filtr",  4, ns=toBitVecs(filtrsz, BITS_INDEX))

s_tgt = dict(s_src)

# Preconditions
preconds = []

# Source program
s_src["output"] = convolution(s_src["image"], s_src["filtr"], preconds)

# Target program
s_tgt["mat"] = convertImageToMatrix(s_tgt["image"], s_tgt["filtr"].ns, preconds)

image_val = s_tgt["image"]
filtr_val = s_tgt["filtr"]
s_tgt["mat2"] = reshape(s_tgt["mat"], [
    image_val.ns[0] * (image_val.ns[1] - filtr_val.ns[0] + 1)
                    * (image_val.ns[2] - filtr_val.ns[1] + 1),
    filtr_val.ns[0] * filtr_val.ns[1] * filtr_val.ns[2]
])

s_tgt["filtr2"] = reshape(s_tgt["filtr"], [
    filtr_val.ns[0] * filtr_val.ns[1] * filtr_val.ns[2],
    filtr_val.ns[3]
])

s_tgt["output"] = matmul(s_tgt["mat2"], s_tgt["filtr2"], preconds)


s = SolverFor("UFBV")

# Goal
s.add(And(preconds))

i_counterex = BitVec("i_counterex", BITS_INDEX)
neg_goal = And(ULT(i_counterex, s_tgt["output"].ns[0] * s_tgt["output"].ns[1]),
                Select(s_src["output"].val, i_counterex) !=
                Select(s_tgt["output"].val, i_counterex))
s.add(neg_goal)


with open("dump.txt", mode='w') as f:
  f.write("\n".join([str(x) for x in preconds]))
  f.write("\n" + str(neg_goal))
with open("dump.smt2", mode='w') as f:
  f.write(s.to_smt2())

# Solve
timeStart = time.time()
result = s.check()
timeEnd = time.time()

def z3ResToStr(result):
  if result == unsat:
    return "CORRECT"
  elif result == sat:
    return "INCORRECT"
  return "UNKNOWN"

print("== Result: %s ==\nRunning time: %s secs" % 
      (z3ResToStr(result), timeEnd - timeStart))
if result == unknown:
  print(s.reason_unknown())
elif result == sat:
  model = s.model()
  print(model)
