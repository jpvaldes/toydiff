## Simple autograd implementation using a tape.
## Look at the runnable examples for basic usage.
import math
import std / sequtils

type
  Variable*[T] = object
    tape*: Tape
    value*: T
    index*: int

  Node* = object
    weights: array[2, float]
    parents: array[2, int]

  Tape* = ref object
    nodes: seq[Node]

  Grad* = object
    derivatives: seq[float]


proc newTape*(): Tape =
  Tape(nodes: newSeq[Node]())

func push0(tape: Tape): int =
  let numNodes = tape.nodes.len
  tape.nodes.add(
    Node(
      weights: [0.0, 0.0],
      parents: [numNodes, numNodes]
    )
  )
  return numNodes

func push1(tape: Tape, index: int, weight: float): int =
  let numNodes = tape.nodes.len
  let node = Node(weights: [weight, 0.0], parents: [index, numNodes])
  tape.nodes.add(node)
  return numNodes

func push2(tape: Tape, parents: array[2, int], weights: array[2, float]): int =
  let numNodes = tape.nodes.len
  let node = Node(weights: weights, parents: parents)
  tape.nodes.add(node)
  return numNodes

func init[T](V: type Variable[T], tape: Tape, value: T, index: int): V = V[T](
  tape: tape,
  value: value,
  index: index
)

func variable*[T](tape: Tape, value: T): Variable[T] =
  ## Init ``Variable`` without a given index.
  ## The index will be set to the number of ``Nodes`` in the ``Tape``.
  runnableExamples:
    var t = newTape()
    let v = t.variable(0.0)
    doAssert v.value == 0.0
    doAssert v.index == 0

  Variable.init(tape, value, tape.push0)

func variable*[T](tape: Tape, value: T, index: int): Variable[T] =
  ## Init ``Variable`` with a given index.
  runnableExamples:
    var t = newTape()
    let v = t.variable(0.0, 1)
    doAssert v.value == 0.0
    doAssert v.index == 1

  Variable.init(tape, value, index)

func `$`*(self: Variable): string =
  result = "Var(value=" & $self.value & ", index=" & $self.index &
      ")"

template wrt*(grad: Grad, self: Variable): float =
  ## Calculate the gradient with respect to the given variable.
  runnableExamples:
    var t = newTape()
    let a = t.variable(3.9)
    let b = t.variable(0.2)
    let sum = a + b
    let sumprime = grad(sum)
    doAssert abs(sumprime.wrt(a) - 1.0) <= 1e-15
    doAssert abs(sumprime.wrt(b) - 1.0) <= 1e-15

  grad.derivatives[self.index]

func grad*[T](self: Variable[T]): Grad =
  let nodes = self.tape.nodes
  var derivs = newSeqWith[float](nodes.len, 0.0)
  derivs[self.index] = 1.0
  for i in countdown(nodes.high, nodes.low):
    for j in 0 ..< 2:
      derivs[nodes[i].parents[j]] += nodes[i].weights[j] * derivs[i]
  Grad(derivatives: derivs)

func `sin`*[T](self: Variable[T]): Variable[T] =
  let value = sin(self.value)
  variable(
    self.tape,
    value,
    self.tape.push1(self.index, cos(self.value))
  )

func `cos`*[T](self: Variable[T]): Variable[T] =
  variable(
    self.tape,
    cos(self.value),
    self.tape.push1(self.index, -sin(self.value))
  )

func `exp`*[T](self: Variable[T]): Variable[T] =
  let value = exp(self.value)
  variable(
    self.tape,
    value,
    self.tape.push1(self.index, value)
  )

func `pow`*[T](self: Variable[T], k: float): Variable[T] =
  variable(
    self.tape,
    pow(self.value, k),
    self.tape.push1(self.index, pow(self.value, k - 1.0))
  )

func `+`*[T](self, other: Variable[T]): Variable[T] =
  ## Sum of two variables

  let value = self.value + other.value
  variable(
    self.tape,
    value,
    self.tape.push2([self.index, other.index], [T(1.0), T(1.0)])
  )

func `+`*[T](self: Variable[T], num: T): Variable[T] =
  ## Ex: x + 1
  let value = self.value + num
  variable(
    self.tape,
    value,
    self.tape.push1(self.index, 1.T)
  )

func `-`*[T](self, other: Variable[T]): Variable[T] =
  ## Substraction
  let value = self.value - other.value
  variable(
    self.tape,
    value,
    self.tape.push2([self.index, other.index], [T(1.0), T(-1.0)])
  )

func `-`*[T](self: Variable[T]): Variable[T] =
  let value = -1.T * self.value
  variable(
    self.tape,
    value,
    self.tape.push1(self.index, -1.T)
  )

func `-`*[T](self: Variable[T], num: T): Variable[T] =
  self + -num

func `*`*[T](self, other: Variable[T]): Variable[T] =
  let value = self.value * other.value
  variable(
    self.tape,
    value,
    self.tape.push2(
      [self.index, other.index],
      [other.value, self.value]
      )
    )

func `*`*[T](coef: T, self: Variable[T]): Variable[T] =
  let value = coef * self.value
  variable(
    self.tape,
    value,
    self.tape.push1(self.index, T(coef))
  )

func `/`*[T](self, other: Variable[T]): Variable[T] =
  if other.value < 1e-15:
    raise newException(DivByZeroDefect, "Division by zero!")
  else:
    let value = self.value / other.value
    variable(
      self.tape,
      value,
      self.tape.push2(
        [self.index, other.index],
        [T(1) / other.value, -value / other.value]
      )
    )

func `/`*[T](self: Variable[T], divisor: T): Variable[T] =
  if divisor < 1e-15:
    raise newException(DivByZeroDefect, "Division by zero!")
  else:
    (1.T / divisor) * self

func `/`*[T](dividend: T, self: Variable[T]): Variable[T] =
  if self.value < 1e-15:
    raise newException(DivByZeroDefect, "Division by zero!")
  else:
    let value = dividend / self.value
    variable(
      self.tape,
      value,
      self.tape.push1(
        self.index,
        -dividend / self.value ^ 2
      )
    )

func sigmoid*[T](self: Variable[T]): Variable[T] =
  ## Sigmoid. Numerically stable implementation.
  ## See `<http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/>`_.
  let value = if self.value >= 0: 1 / (1 + exp(-self.value))
              else: exp(self.value) / (1 + exp(self.value))
  variable(
    self.tape,
    value,
    self.tape.push1(self.index, value * (1 - value))
  )

func relu*[T](self: Variable[T], thr: float = 0.0): Variable[T] =
  variable(
    self.tape,
    if self.value > thr: self.value else: T(0),
    self.tape.push1(self.index, if self.value > thr: T(1) else: T(0))
  )

func tanh*[T](self: Variable[T]): Variable[T] =
  ## Hyperbolic tangent.
  ## Related to ``sigmoid``: tanh = 2 * sigmoid(2 * value) - 1
  2.T * sigmoid(self.tape.variable(2.T) * self) - 1.T

when isMainModule:
  block simple_test:
    var tape = newTape()
    let
      x = tape.variable(0.5)
      y = tape.variable(4.2)
      z = x * y + x.sin
      g = grad(z)
    doAssert abs(z.value - 2.579425538604203) <= 1e-15
    doAssert abs(g.wrt(x) - (y.value + cos(x.value))) <= 1e-15
    doAssert abs(g.wrt(y) - x.value) <= 1e-15
