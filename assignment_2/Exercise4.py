import numpy as np

from typing import Callable, Tuple



def exhaustive_search(
  f: Callable[[float], float], 
  a: float, 
  b: float, 
  tol: float=1e-1
) -> Tuple[float]:
  N = int(2*(b-a)/tol-1) # tolerance to number of points
  # generate N evenly spaced points on the interval [a,b]
  points = np.linspace(a, b, N)

  start, finish = 0, N
  for n in range(N): # stopping criterion 1
    if abs(b-a) <= tol: # stopping criterion 2
      break
      
    if f(a) <= f(b): # reduce the interval based on unimodality
      finish -= 1
      b = points[finish]
    else:
      start += 1
      a = points[start]

  return (a,b)


def dichotomous_search(
  f: Callable[[float], float], 
  a: float, 
  b: float, 
  tol: float=1e-3
) -> Tuple[float]:
  # Ln = L0/(2^(0.5*n))+delta*(1 - 1/(2^(0.5*n))) = tol
  # n = 2*ln((L0-delta)/(tol-delta))/ln(2)
  # ref: Engineering Optimization by Singiresu S. Rao
  delta = 1e-2
  N = int(2*np.log((abs((b-a)-delta))/abs((tol-delta)))/np.log(2))
  print(N)
  for n in range(N): # stopping criterion 1
    if abs(b-a) <= tol: # stopping criterion 2
      break

    c = (a+b) / 2 - delta
    d = (a+b) / 2 + delta
    print(c, d, f(c), f(d))
    if f(c) < f(d):
      b = c
    else:
      a = d
    print(n, a, b)
  return (a,b)


def interval_halving(
  f: Callable[[float], float], 
  a: float, 
  b: float, 
  tol: float=1e-3
) -> Tuple[float]:
  # Ln = L0 * 0.5^(0.5*(n-1)) = tol
  # n = 2 * ln(tol/L0) / ln(0.5) + 1
  # ref: Engineering Optimization by Singiresu S. Rao
  N = int(2*np.log(abs(tol/float(b-a)))/np.log(0.5)+1)
  print(N)
  for n in range(N): # stopping criterion 1
    if abs(b-a) <= tol: # stopping criterion 2
      break
    # generate three evenly spaced points on the interval [a,b]  
    points = np.linspace(a,b,3)
    x0, x1, x2 = points[1], points[0], points[2]
  
    f0, f1, f2 = f(x0), f(x1), f(x2)
    print(x0, x1,x2, f0,f1,f2)
    if f2 > f0 > f1:
      b, x0 = x0, x1
    elif f2 < f0 < f1:
      a, x0 = x0, x2
    else:
      # in this case optimum is probably bracketed by current a and b
      # and the limits will not change, which is not good
      # as the algorithm will loop (at most) N times without any changes
      # hence the limits are "forced" to decrease by a small value on both ends
      a, b = x1 + tol, x2 - tol
    print(n, a, b)
  return (a,b)
