from typing import Callable, Tuple



def unrestricted_search(
  f: Callable[[float], float],
  a: float,
  b: float,
  fixed: bool=True,
  delta: float=1e-2,
  gamma: float=2.0,
  tol: float=1e-3,
  N: int=20
) -> Tuple[float]:
  a1 = a
  for n in range(N): # stopping criterion 1
    if  abs(b-a) <= tol: # stopping criterion 2
      break
    # in non-uniform step-size, steps are increasing if
    # they are made in the same direction hence prev delta is kept 
    old_delta = delta 
    a, a1 = a1 + delta, a # make a step  
    if f(a) > f(a1): # check moving direction based on unimodality
      delta = abs(delta)
    else: # reverse the direction of the search
      delta = -delta
    # unrestricted search is implemented w/ a fixed step-size
    # an improved version consists of doubling the step size
    # as long as the step results in an improvement
    # ref: Engineering Optimization by Singiresu S. Rao
    if not fixed and old_delta * delta > 0:
      delta = float(gamma*delta)  
    
  return (a,b) if a <= b else (b,a)
