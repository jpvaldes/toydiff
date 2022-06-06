import ../../src/toydiff
import math

block create_variable:
  var
    tape = newTape()
    v = tape.variable(1.0)
  doAssert v.value == 1.0
  doAssert v.index == 0

block sum:
  var
    tape = newTape()
    v = tape.variable(0.5)
    w = tape.variable(1.0)
    x = v + w
    g = grad(x)

  doAssert x.value == 1.5
  doAssert x.index == 2
  doAssert abs(g.wrt(v) - 1.0) < 1e-15
  doAssert abs(g.wrt(w) - 1.0) < 1e-15

block unary_minus:
  var tape = newTape()
  let
    x = tape.variable(2.0)
    y = -x
    g = grad(y)

  doAssert abs(y.value + 2.0) < 1e-15
  doAssert abs(g.wrt(x) + 1.0) < 1e-15

block minus_num:
  var tape = newTape()
  let
    x = tape.variable(2.0)
    num = 1.0
    y = x - num
    g = grad(y)

  doAssert abs(y.value - 1.0) < 1e-15
  doAssert abs(g.wrt(x) - 1.0) < 1e-15

block poly:
  var
    tape = newTape()
  let
    x = tape.variable(1.0)
    y = x * x - 2.0 * x + tape.variable(6.0)
    g = grad(y)

  doAssert y.value == 5.0
  doAssert g.wrt(x) == 0.0

block trig:
  var
    tape = newTape()
  let
    x = tape.variable(PI)
    y = 2.0 * sin(x)
    g = grad(y)

  doAssert abs(y.value - 0.0) < 1e-15
  doAssert g.wrt(x) == -2.0

block mul:
  var tape = newTape()
  let
    x = tape.variable(2.0)
    y = tape.variable(3.0)
    res = x * y
    resprime = grad(res)

  doAssert abs(x.value * y.value - 6.0) < 1e-15
  doAssert abs(resprime.wrt(x) - 3.0) < 1e-15
  doAssert abs(resprime.wrt(y) - 2.0) < 1e-15

block mul_single_var:
  var tape = newTape()
  let
    x = tape.variable(2.0)
    coeff = 1.5
    res = coeff * x
    g = grad(res)

  doAssert abs(res.value - 3.0) < 1e-15
  doAssert abs(g.wrt(x) - 1.5) < 1e-15

block division:
  var tape = newTape()
  let
    x = tape.variable(2.0)
    y = tape.variable(1.0)
    res = x / y
    g = grad(res)

  doAssert abs(res.value - 2.0) < 1e-15
  doAssert abs(g.wrt(x) - 1.0) < 1e-15
  doAssert abs(g.wrt(y) + 2.0) < 1e-15

block division_single_var:
  var tape = newTape()
  let
    x = tape.variable(2.0)
    y = 1.0
    res = x / y
    g = grad(res)

  doAssert abs(res.value - 2.0) < 1e-15
  doAssert abs(g.wrt(x) - 1.0) < 1e-15

block single_var_division:
  var tape = newTape()
  let
    x = tape.variable(3.0)
    numerator = 9.0
    res = numerator / x
    g = grad(res)

  doAssert abs(res.value - 3.0) < 1e-15
  doAssert abs(g.wrt(x) + 1.0) < 1e-15

block sigmoidal:
  var tape = newTape()
  let
    x = tape.variable(0.5)
    s = sigmoid(x)
    g = grad(s)

  doAssert abs(s.value - 0.622459331201855) < 1e-15
  doAssert abs(g.wrt(x) - 0.235003712201594) < 1e-15

block relu:
  var tape = newTape()
  let
    x = tape.variable(2.0)
    r = relu(x)
    g = grad(r)

  doAssert abs(r.value - 2.0) < 1e-15
  doAssert abs(g.wrt(x) - 1.0) < 1e-15

block relu_thr:
  var tape = newTape()
  let
    x = tape.variable(1.0)
    r = relu(x, 2.0)
    g = grad(r)

  doAssert abs(r.value - 0.0) < 1e-15
  doAssert abs(g.wrt(x) - 0.0) < 1e-15

block tanh:
  var tape = newTape()
  let
    x = tape.variable(2.0)
    t = tanh(x)
    g = grad(t)

  echo g.wrt(x)
  doAssert abs(t.value - 0.964027580075817) < 1e-15
  doAssert abs(g.wrt(x) - 0.0706508248531644) < 1e-15
