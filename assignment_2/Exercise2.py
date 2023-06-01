import numpy as np

from typing import Callable, List, Tuple


def golden_section_search(
  f: Callable[[float], float],
  a: float,
  b: float,
  tol: float=1e-3
) -> Tuple[float]:
  """
  reduces an initial interval [a,b] using the golden section search method.

  parameters
  ----------
  f: function
    function f(x) continuous on [a,b] for which f(x)=0 is calculated.
  a: real number
    lower limit in the initial bracketing interval [a,b] : a < b.
  b: real number
    upper limit in the initial bracketing interval [a,b] : b > a.
  tol: tolerance
    stopping criterion : |b_n-a_n| <= tolerance.

  returns
  -------
  a,b: real numbers
    reduced interval, where a and b are lower- and upper-limits, respectively.

  examples
  --------
  >>> f = lambda x: -x**2+21.6*x+3
  >>> golden_section_search(f, 0, 20)
  (10.79962481733144, 10.80044194431161)

  """
  
  phi = (np.sqrt(5)-1) / 2 # golden ratio
  # Ln = phi^(n-1) * L0 = tol
  # n = ln(tol/L0)/ln(phi) + 1
  # ref: Engineering Optimization by Singiresu S. Rao
  N = int(np.log(abs(tol/float(b-a)))/np.log(phi)+1)

  c = a + (b-a) * (1-phi)
  d = a + (b-a) * phi
  for n in range(N): # stopping criterion 1
    if abs(b-a) <= tol: # stopping criterion 2
      break

    if f(c) > f(d): # <= for max, > for min
      a, c = c, d
      d = a + (b-a) * phi
    else:
      b, d = d, c
      c = a + (b-a) * (1-phi)

  return (a,b)


def fibonacci_sequence(N: int) -> List[int]:
  """
  returns N-first Fibonacci numbers, excluding zero.
  
  parameters
  ----------
  N: (positive) integer
    number of the first N Fibonacci elements to be calculated.

  returns
  -------
  F: list of (positive) integers
    Fibonacci sequence calculated to N first elements.

  examples
  --------
  >>> fibonacci_sequence(10)
  [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]

  """

  F = []
  a, b = 0, 1
  for i in range(N):
    F.append(b)
    a, b = b, a+b

  return F


def fibonacci_search(
  f: Callable[[float], float],
  a: float,
  b: float,
  tol: float=1e-3
) -> Tuple[float]:
  """
  reduces an initial interval [a,b] using the fibonacci search method.

  parameters
  ----------
  f: function
    function f(x) continuous on [a,b] for which f(x)=0 is calculated.
  a: real number
    lower limit in the initial bracketing interval [a,b] : a < b.
  b: real number
    upper limit in the initial bracketing interval [a,b] : b > a.
  tol: tolerance
    stopping criterion : |b_n-a_n| <= tolerance.

  returns
  -------
  a,b: real numbers
    reduced interval, where a and b are lower- and upper-limits, respectively.

  examples
  --------
  >>> f = lambda x: -x**2+21.6*x+3
  >>> fibonacci_search(f, 0, 20)
  (10.799455630386989, 10.800153540147257)

  """
  
  # define arbitrary large fibonacci sequence beforehand
  K = 100
  F = fibonacci_sequence(K)
  # find the smallest fibonacci number : Fn >= (b-a) / tolerance 
  Fn = F[min(range(K), key=lambda k: abs(F[k]-(b-a)/tol))]
  N = F.index(Fn) + 1 # +1 to make the inequality strict
  # initial two experiments
  p = a + F[N-2] * (b-a) / F[N]
  q = a + F[N-1] * (b-a) / F[N]
  K = N if N < K else K # update the K used in the loop based on the value of N
  for k in range(K): # stopping criterion 1
    if abs(b-a) <= tol: # stopping criterion 2
      break

    print(k, a, b)
    if f(p) > f(q): # <= for max, > for min
      a, p, N = p, q, N-1
      q = a + F[N-1] * (b-a) / F[N]
    else:
      b, q, N = q, p, N-1
      p = a + F[N-2] * (b-a) / F[N]

  return (a,b)
