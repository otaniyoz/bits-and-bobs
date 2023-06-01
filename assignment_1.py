import numpy as np
import matplotlib.pyplot as plt

from itertools import zip_longest
from typing import Callable, List, Union, Dict


# define custom color scheme for plotting 
PALETTE = {
  "light":(0.96, 0.96, 0.95), "dark":(0.1, 0.1, 0.1), "grey":(0.26, 0.27, 0.35), 
  "cyan":(0.54, 0.91, 0.99), "blue":(0.38, 0.44, 0.64), "green":(0.31, 0.98, 0.48), 
  "pink":(1, 0.47, 0.78), "purple":(0.74, 0.57,0.97), "orange":(1, 0.30, 0), 
  "red":(1, 0.33, 0.33), "yellow":(0.94, 0.98, 0.54)
}
# define custom plotting style
plt.rcParams["figure.figsize"] = (12, 6)

plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.fontsize"] = 16
plt.rcParams["legend.title_fontsize"] = 16
plt.rcParams["legend.edgecolor"] = PALETTE["dark"]

plt.rcParams["text.color"] = PALETTE["dark"]
plt.rcParams["font.monospace"] = "Ubuntu Mono"
plt.rcParams["font.family"] = "monospace"

plt.rcParams["lines.linewidth"] = 3
plt.rcParams["lines.markersize"] = 6

plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in" 
plt.rcParams["xtick.color"] = PALETTE["dark"]
plt.rcParams["ytick.color"] = PALETTE["dark"]

plt.rcParams["axes.titlesize"] = 20
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelcolor"] = PALETTE["dark"]
plt.rcParams["axes.facecolor"] = PALETTE["light"]
plt.rcParams["axes.edgecolor"] = PALETTE["dark"]


def bisection(
  f: Callable[[float],float],
  a: float,
  b: float,
  tol: float,
  opt: bool=True,
  N: int=40
) -> Union[List[float], float]:
  """
  given a function f(x) continuous on an interval [a,b] and f(a)*f(b)<0,
  solves f(x)=0 via bisection method for a given tolerance or N iterations.

  parameters
  ----------
  f: function
    function, for which f(x)=0 is calculated.
  a: real number
    lower limit in the initial bracketing interval [a,b] : a < b.
  b: real number
    upper limit in the initial bracketing interval [a,b] : b > a.
  tol: tolerance
    stopping criterion : |b_n-a_n| <= tolerance.
  opt: boolean
    if True, only last calculated midpoint is returned.
  N: positive integer
    number of maximum iterations.

  returns
  -------
  m: real number
    last caculated midpoint.
  history: list of real numbers
    list of all calculated midpoints.

  examples
  --------
  >>> f = lambda c: 668.06 * (1-np.exp(-0.146843*c)) / c - 40
  >>> a, b = 12, 16
  >>> machine_epsilon = 2**-52
  >>> tolerance = np.sqrt(machine_epsilon) * (abs(a)+abs(b)) / 2
  >>> bisection(f,a,b,tolerance)
  |14.801109969615936

  >>> f = lambda x: np.power(x,10) - 1
  >>> a, b = 0, 1.3
  >>> machine_epsilon = 2**-52
  >>> tolerance = np.sqrt(machine_epsilon) * (abs(a)+abs(b)) / 2
  >>> bisection(f,a,b,tolerance)
  1.0000000009313226
  """

  history = []
  for n in range(N): # stopping criterion 1
    # make sure the root is bracketed, otherwise throw an error
    assert f(a) * f(b) < 0, "Root is not bracketed."

    m = (a+b) / 2
    history.append(m)
    if abs(b-a) <= tol: # stopping criterion 2
      break
    # update interval based on unimodality conditions
    if f(a) * f(m) < 0: 
      a, b = a, m # pythonic swap
    elif f(b) * f(m) < 0: 
      a, b = m, b 

  return (m if opt else history)


def regula_falsi(
  f:Callable[[float],float],
  a:float,
  b:float,
  tol:float,
  opt:bool=True,
  N:int=30
) -> Union[List[float], float]:
  """
  given a function f(x) continuous on an interval [a,b] and f(a)*f(b)<0,
  solves f(x)=0 via regula falsi method for a given tolerance or N iterations.

  parameters
  ----------
  f: function
    function for which f(x)=0 is calculated.
  a: real number
    lower limit in the initial bracketing interval [a,b] : a < b.
  b: real number
    upper limit in the initial bracketing interval [a,b] : b > a.
  tol: tolerance
    stopping criterion : |b_n-a_n| <= tolerance.
  opt: boolean
    if True, only last calculated midpoint is returned.
  N: positive integer
    number of maximum iterations.

  returns
  -------
  c: real number
    last caculated x-intercept.
  history: list of real numbers
    list of all calculated x-intercepts.

  examples
  --------
  >>> f = lambda c: 668.06 * (1-np.exp(-0.146843*c)) / c - 40
  >>> a, b = 12, 16
  >>> machine_epsilon = 2**-52
  >>> tolerance = np.sqrt(machine_epsilon) * (abs(a)+abs(b)) / 2
  >>> regula_falsi(f,a,b,tolerance)
  14.801109969022702

  >>> f = lambda x: np.power(x,10) - 1
  >>> a, b = 0, 1.3
  >>> machine_epsilon = 2**-52
  >>> tolerance = np.sqrt(machine_epsilon) * (abs(a)+abs(b)) / 2
  >>> regula_falsi(f,a,b,tolerance)
  0.9965725090161869
  """

  history = []
  for n in range(N): # stopping criterion 1
    # make sure the root is bracketed, otherwise throw an error
    assert f(a) * f(b) < 0, "Root is not bracketed."
    if abs(b-a) <= tol: # stopping criterion 2
      break

    c = (a*f(b)-b*f(a)) / (f(b)-f(a))
    # update intercal based on unimodality conditions
    if f(a) * f(c) < 0: 
      a, b = a, c
    elif f(b) * f(c) < 0: 
      a, b = c, b
    history.append(c)

  return (c if opt else history)


def secant(
  f: Callable[[float],float], 
  a: float,
  b: float,
  tol: float,
  opt: bool=True,
  N: int=30
) -> Union[List[float],float]:
  """
  solves f(x)=0 on a closed interval [a,b] using secant method
  for a given tolerance or up to N iterations.

  parameters
  ----------
  f: function
    function for which f(x)=0 is calculated.
  a: real number
    lower limit in the initial bracketing interval [a,b] : a < b.
  b: real number
    upper limit in the initial bracketing interval [a,b] : b > a.
  tol: tolerance
    stopping criterion : |b_n-a_n| <= tolerance.
  opt: boolean
    if True, only last calculated root is returned.
  N: positive integer
    number of maximum iterations.

  returns
  -------
  c: real number
    last caculated root of secant line.
  history: list of real numbers
    list of all calculated roots of secant lines.

  examples
  --------
  >>> f = lambda c: 668.06 * (1-np.exp(-0.146843*c)) / c - 40
  >>> a, b = 12, 16
  >>> machine_epsilon = 2**-52
  >>> tolerance = np.sqrt(machine_epsilon) * (abs(a)+abs(b)) / 2
  >>> secant(f,a,b,tolerance)
  14.801109969022695

  >>> f = lambda x: np.power(x,10) - 1
  >>> a, b = 0, 1.3
  >>> machine_epsilon = 2**-52
  >>> tolerance = np.sqrt(machine_epsilon) * (abs(a)+abs(b)) / 2
  >>> secant(f,a,b,tolerance)
  0.1817588726989925 # does not converge
  """

  history = []
  for n in range(N): # stopping criterion 1
    fa, fb = f(a), f(b)

    if abs(b-a) <= tol: # stopping criterion 2
      break
    # rewrite f(b) * (b-a) / (f(b)-f(a)) as f(b) / (f(b)-f(a)) / (b-a) 
    # in an attempt to avoid possible division by zero
    # probably has no effect or even leads to loss in precision
    c = b - fb / (float(fb-fa)/(b-a))
    a, b = b, c
    history.append(c)

  return (c if opt else history)


def newton(
  f: Callable[[float],float],
  df: Callable[[float],float],
  x0: float,
  tol: float,
  opt: bool=True,
  N: int=20
) -> Union[List[float],float]:
  """
  solves f(x)=0 using newton method
  for a given tolerance or up to N iterations.

  parameters
  ----------
  f: function
    function for which f(x)=0 is calculated.
  df: first derivative
    derivative of function f(x).
  x0: real number
    initial guess for a root.
  tol: tolerance
    stopping criterion : |f(x_n)/f'(x_n)| <= tolerance.
  opt: boolean
    if True, only last calculated root is returned.
  N: positive integer
    number of maximum iterations.

  returns
  -------
  xn: real number
    last caculated root.
  history: list of real numbers
    list of all calculated roots.

  examples
  --------
  >>> f = lambda c: 668.06 * (1-np.exp(-0.146843*c)) / c - 40
  >>> df = lambda c: np.exp(-0.146843*c)*(98.0999*c-668.06*np.exp(0.146843*c)+668.06)/np.power(c,2)
  >>> a, b = 12, 16
  >>> machine_epsilon = 2**-52
  >>> tolerance = np.sqrt(machine_epsilon) * (abs(a)+abs(b)) / 2
  >>> newton(f,df,a,tolerance)
  14.801109969022702

  >>> f = lambda x: np.power(x,10) - 1
  >>> df = lambda x: 10 * np.power(x,9)
  >>> a, b = 0, 1.3
  >>> machine_epsilon = 2**-52
  >>> tolerance = np.sqrt(machine_epsilon) * (abs(a)+abs(b)) / 2
  >>> newton(f,df,b,tolerance)
  1.0
  """

  xn = x0
  history = []
  for n in range(N): # stopping criterion 1
    fxn, dfxn = f(xn), df(xn)
    # avoid numerical "blow-up"
    assert dfxn != 0, "f'({0}) is zero!".format(xn)
    d = fxn / dfxn # store it for later use
    xn -= d
    history.append(xn) 
    if abs(d) <= tol: # stopping criterion 2
      break

  return (xn if opt else history)


def p(roots:List[float], x:float=None, n:int=2)->Union[float,None]:
  """ approximates order of convergence p """

  if len(roots) >= n + 2: # inter- and exterpolation if enough data
    a = np.log2(abs((roots[n+1]-roots[n])/(roots[n]-roots[n-1])))
    b = np.log2(abs((roots[n]-roots[n-1])/(roots[n-1]-roots[n-2])))
  elif len(roots) < n + 2 and x: # otherwise try using true root
    a = np.log2(abs(roots[-1]-x))
    b = np.log2(abs(roots[-2]-x))
  else:
    return None

  return round(a/b, 2)


def plot_root_finding(
  f: Callable[[float],float],
  a: float,
  b: float,
  methods: List[Dict[str, Union[List[float], str]]],
  x_true: float
):
  """ plots different root finding methods on the same graph """

  xs = np.linspace(a,b,int(abs(b-a))*50)
  fig, ax = plt.subplots()
  ax.plot(xs,f(xs),
          color=PALETTE["blue"],
          zorder=1)
  for method in methods:
    for x in method["values"]:
      ax.scatter(x,f(x),
                 color=method["color"],
                 label=method["name"],
                 alpha=0.7,
                 zorder=2)
  plt.vlines(x_true,f(a)-1,f(b), 
             color=PALETTE["grey"], 
             linestyle="--",
             zorder=1,
             label="Line through solution")
  handles, labels = ax.get_legend_handles_labels()
  by_label = dict(zip(labels, handles))
  plt.legend(by_label.values(), by_label.keys())
  ax.set_xlabel("x")
  ax.set_ylabel("f(x)")
  ax.set_xlim(a,b)
  ax.set_ylim(f(a)-1,f(b))
  plt.show()


def tabulate(values:List[float], header:List[str]=None):
  """ prints values in tabular form following markdown format """

  format_row = "{:<20}|" * (len(values)+1)
  if header: 
    print(format_row.format(*header))
    print((":" + 19*"-" + "|")*len(header))
  for i,row in enumerate(list(zip_longest(*values, fillvalue=""))):
    print(format_row.format(i, *row))


def solution(f,df,a,b,x):
  """ 
  helper function to solve assignment. it finds roots for given functions
  using implemented methods, then tabulates root-finding process, 
  calculates practical order of convergence, and makes plots.
  """

  machine_epsilon=2**-52  
  tolerance=np.sqrt(machine_epsilon)*(abs(a)+abs(b))/2
  bi_res=bisection(f,a,b,tolerance,False)  
  rf_res=regula_falsi(f,a,b,tolerance,False)
  sec_res=secant(f,a,b,tolerance,False)  
  new_res=newton(f,df,b,tolerance,False)
  header=["Iteration", "Bisection", "Regula Falsi", "Secant", "Newton"]
  values=[bi_res,rf_res,sec_res,new_res]
  print("Bisection order of convergence, 𝑝≈{0}".format(p(bi_res,x)))
  print("Regula Falsi order of convergence, 𝑝≈{0}".format(p(rf_res,x)))
  print("Secant order of convergence, 𝑝≈{0}".format(p(sec_res,x)))
  print("Newton order of convergence, 𝑝≈{0}".format(p(new_res,x)))
  tabulate(values,header)
  methods = [
    dict(
      name="Bisection",
      color=PALETTE["red"],
      values=bi_res
    ),
    dict(
      name="Regula Falsi",
      color=PALETTE["orange"],
      values=rf_res
    ),  
    dict(
      name="Secant",
      color=PALETTE["purple"],
      values=sec_res
    ),
    dict(
      name="Newton",
      color=PALETTE["green"],
      values=new_res
    ),  
  ]
  plot_root_finding(f,a,b,methods,x)  


def main():
  f1 = lambda c: 668.06 * (1-np.exp(-0.146843*c)) / c - 40
  df1 = lambda c: np.exp(-0.146843*c)*(98.0999*c-668.06*np.exp(0.146843*c)+668.06)/np.power(c,2)
  a1,b1,x1=12,16,14.8011099690227
  solution(f1,df1,a1,b1,x1)

  f2 = lambda x: np.power(x,10) - 1
  df2 = lambda x: 10 * np.power(x,9)
  a2,b2,x2=0,1.3,1.0
  solution(f2,df2,a2,b2,x2)

  f3 = lambda x: np.power(x,2) - 3
  df3 = lambda x: 2 * x
  a3,b3,x3=-6,6,1.4
  solution(f3,df3,a3,b3,x3)


if __name__=="__main__":
  main()
