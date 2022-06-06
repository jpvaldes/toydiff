import ../../src/toydiff
import math

block simple_nn:
  # Example in
  # https://karpathy.github.io/neuralnets/
  # Basically, the value of ``s`` should increase after a step of `gradient
  # descent` with gradients computed with reverse mode auto-differentiation.
  var
    tape = newTape()
  let
    a = tape.variable(1.0)
    b = tape.variable(2.0)
    c = tape.variable(-3.0)
    x = tape.variable(-1.0)
    y = tape.variable(3.0)

  func forwardNeuron[T](a, b, c, x, y: Variable[T]): Variable[T] =
    let
      ax = a * x
      by = b * y
      axpby = ax + by
      axpbypc = axpby + c
    axpbypc.sigmoid

  let s = forwardNeuron(a, b, c, x, y)

  let
    sprime = s.grad
    step = 0.01
    a2 = tape.variable(a.value + (step * sprime.wrt(a)))
    b2 = tape.variable(b.value + (step * sprime.wrt(b)))
    c2 = tape.variable(c.value + (step * sprime.wrt(c)))
    x2 = tape.variable(x.value + (step * sprime.wrt(x)))
    y2 = tape.variable(y.value + (step * sprime.wrt(y)))

  let s2 = forwardNeuron(a2, b2, c2, x2, y2)

  doAssert abs(s.value - 0.8807970779778823) < 1e-15
  doAssert abs(s2.value - 0.8825501816218984) < 1e-15
