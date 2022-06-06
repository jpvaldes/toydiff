# Toydiff: Nim simple autodiff

This code is just for educational purposes and not meant to be practical in any
way. There are plenty of good autodiff libraries out there that you can use for
actual projects like Tensorflow, Pytorch, Jax, or Arraymancer.

It is based on the following blog post:
https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation.
There's also a similar [archived project](https://github.com/mratsim/nim-rmad)
by the creator of Arraymancer in Nim.

At the time of writing the current stable [Nim](https://nim-lang.org) branch
is 1.6.

To compile `toydiff.nim`, do:

```sh
nim cbuild src/toydiff.nim
```

To run the tests, use:

```sh
testament pattern tests/*
```
