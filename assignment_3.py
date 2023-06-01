import numpy as np
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")


def quadratic_interpolation(f,x1,x2,x3,tol=1e-6,max_iter=100):
  A = (x1-x2)*(x1-x3)
  B = (x2-x1)*(x2-x3)
  C = (x3-x1)*(x3-x2)
  x4 = (f(x1)/A*(x2+x3)+f(x2)/B*(x1+x3)+f(x3)/C*(x1+x2))/(2*(f(x1)/A+f(x2)/B+f(x3)/C)) 

  iter=0
  print(f"starting quadratic interpolation")
  print(f"initial points are ({x1:.4f},{x2:.4f},{x3:.4f})")
  while np.abs(x1-x3) > tol and iter < max_iter:
    if x4 > x2:
      if f(x4) >= f(x2): 
        x3 = x4
      else:
        x1, x2 = x2, x4
    elif x4 < x2:
      if f(x4) >= f(x2): 
        x1 = x4
      else:
        x3, x2 = x2, x4
    iter += 1
  print(f"after {iter} iterations the method converged to")   
  print(f"({x1:.4f},{x2:.4f},{x3:.4f})")

  q = lambda x: (f(x1) * (x - x2) * (x - x3) / ((x1 - x2) * (x1 - x3))
                 + f(x2) * (x - x1) * (x - x3) / ((x2 - x1) * (x2 - x3))
                 + f(x3) * (x - x1) * (x - x2) / ((x3 - x1) * (x3 - x2)))
  
  return q, x4


def cubic_interpolation(f,df,x1,x2,tol=1e-6,max_iter=100):
  iter=0
  print(f"starting cubic interpolation")
  print(f"initial points are ({x1:.4f},{x2:.4f})")
  while np.abs(x1-x2) > tol and iter < max_iter:
    a = df(x2) + df(x1) - 2*(f(x2)-f(x1))
    b = 3*(f(x2)-f(x1)) - df(x2) - 2*df(x1)
    e = df(x1)/(-b - np.sqrt(np.power(b,2) - 3*a*df(x1)))
    if b < 0:
      e = (-b + np.sqrt(np.power(b,2) - 3*a*df(x1)))/(3*a)
    x = x1 + (x2 - x1)*e
    if np.sign(df(x)) == np.sign(df(x1)):
      x1 = x
    else:
      x2 = x
    iter += 1
  print(f"after {iter} iterations the method converged to:")   
  print(f"({x1:.4f},{x2:.4f})")

  c = lambda x: (a*np.power((x - x1)/(x2 - x1),3)
                 + b*np.power((x - x1)/(x2 - x1),2)
                 + df(x1)*(x - x1) + f(x1))

  return c, x


def bisection(f, a, b, tol=1e-3, opt=True, N=100):
  history = []
  for n in range(N): # stopping criterion 1
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


def main(make_plots=False):
  # no real roots
  f1 = lambda x: x + 1/x
  # 0.26, 2.54
  f2 = lambda x: np.exp(x) - 5*x
  # -2.79, 0.25, 2.73
  f3 = lambda x: np.power(x,5) - 5*np.power(x,3) - 20*x + 5
  # -1, 0.5, 0.75
  f4 = lambda x: 8*np.power(x,3) - 2*np.power(x,2) - 7*x + 3
  # no real roots
  f5 = lambda x: 5 + np.power(x-2,6)
  # 1, 1.00
  f6 = lambda x: (100*np.power(1-np.power(x,3),2) + 1 - np.power(x,2)
                  + 2*np.power(1-x,2))
  # -0.00999, 0.0000990
  f7 = lambda x: np.exp(x) - 2*x + 0.01/x - 1e-6/np.power(x,2)

  df1 = lambda x: 1 - x**(-2)
  df2 = lambda x: np.exp(x) - 5
  df3 = lambda x: 5*np.power(x,4) - 15*np.power(x,2) - 20
  df4 = lambda x: 24*np.power(x,2) - 4*x - 7
  df5 = lambda x: 6*np.power(x-2,5)
  df6 = lambda x: (3-x)/(100*(-1+x)*np.power(1+x+np.power(x,2),2))
  df7 = lambda x: 2*np.exp(-x)*(5e-7-0.005*x+np.power(x,3))/np.power(x,2) 

  fs = [f1, f2, f3, f4, f5, f6, f7]
  dfs = [df1, df2, df3, df4, df5, df6, df7]

  x1, x3 = 0.01,4 # using asymetric bounds to avoid numerical errors
  x2 = np.mean([x1,x3])
  plotting_range = np.linspace(x1,x3,500)
  for idx,s in enumerate(zip(fs,dfs)):
    print(10*"=")
    f, df = s[0], s[1]
    q, x_star = quadratic_interpolation(f,x1,x2,x3)
    print(f"x*={x_star:.4f}\n")
    c, x_star = cubic_interpolation(f,df,x1,x3)
    print(f"x*={x_star:.4f}\n")

    if make_plots:
      plt.plot(plotting_range, f(plotting_range), label="f(x)")
      plt.plot(plotting_range, q(plotting_range), label="q(x)", linestyle=(0, (1,5)))
      plt.plot(plotting_range, c(plotting_range), label="c(x)", linestyle="dotted")
      plt.legend(loc="upper right")
      plt.ylim(-100,200)
      plt.xlim(x1,x3)
      plt.savefig(f"part{idx}.png", dpi=200, bbox_inches="tight", transparent=True)
      plt.close()


if __name__ == "__main__":
  main()
