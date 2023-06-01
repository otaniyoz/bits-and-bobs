import itertools
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")


def clip_value(x, x_min, x_max):
  x_clipped = x
  if x >= x_max:
    x_clipped = x_max
  elif x <= x_min:
    x_clipped = x_min
  return x_clipped


def lambda_iteration(
  p1_func, p2_func, p3_func, 
  pd=450.0, l_l=7.0, l_h=8.0, tol=1e-3
):
  iter = 1
  print("lambda iteration")
  print(f"{'iter':>4} {'lambda':>8} {'total gen':>10} {'p1':>10} {'p2':>10} {'p3':>10}")
  while abs(l_h-l_l) > tol and iter <= 10:
    l_m = round((l_h+l_l)/2,4)
    p1 = clip_value(round(p1_func(l_m),4), 45, 350)
    p2 = clip_value(round(p2_func(l_m),4), 45, 350)
    p3 = clip_value(round(p3_func(l_m),4), 47.5, 450)
    s = round(sum([p1,p2,p3]),4)
    assert sum([p1_func(l_l),p2_func(l_l),p3_func(l_l)]) - pd < 0
    assert sum([p1_func(l_h),p2_func(l_h),p3_func(l_h)]) - pd > 0
    print(f"{iter:4} {l_m:8} {s:10} {p1:10} {p2:10} {p3:10}")
    if s - pd > 0:
      l_h = l_m
    else: 
      l_l = l_m
    iter += 1
  print()
  return p1, p2, p3


def base_point_participation_factor(coefs, base_values, pd_new=495):
  new_values = []
  demand_diff = pd_new - base_values[-1]
  denominator = sum([round(float(1/c),4) for c in coefs])
  print("base point and participation factor")
  print("cost function second derivative: ", coefs)
  print("base values: ", base_values)
  print("denominator: ", denominator)
  print("demand difference: ", demand_diff)
  print(f"{'base point':>10} {'participation factor':>20} {'new value':>10}")
  for i in range(len(base_values)-1): 
    base_point = base_values[i]
    participation_factor = round(float(1/coefs[i]) / denominator,4) 
    new_value = round(base_point+participation_factor*demand_diff,4)
    print(f"{base_point:>10} {participation_factor:>20} {new_value:>10}")
    new_values.append(new_value) 
  print()
  return new_values 


def matrix_product(A, B):
  print("multiplying matrices: AB=C")
  print("A", A)
  print("B", B)
  C = [] 
  m, n, p = len(A), len(B), len(B[0])
  for i in range(m): # row
    c_i = []
    for j in range(p): # column
      c_ij = 0 
      for k in range(n):
        c_ij += A[i][k]*B[k][j]
      c_i.append(c_ij)
    C.append(c_i)
  print("C", C)
  return C


def argmin(l):
  return l.index(min(l))


def signature(sigma):
  parity = 1
  n = len(sigma)
  s = list(sigma[:])
  for i in range(n):
    if s[i] != i:
      parity *= -1
      min_idx = argmin(s[i:]) + i
      s[i], s[min_idx] = s[min_idx], s[i]
  print(f"sgn({sigma}) = {parity:>2}") 
  return parity


# leibniz formula 
def matrix_determinant(A):
  det = 0
  n = len(A)
  Sn = list(itertools.permutations(list(range(n)), n))
  print(f"permutations (#S{n}={len(Sn)}): ", Sn)
  for s in Sn:
    product = 1
    for i in range(n): 
      product *= A[s[i]][i] 
    det += signature(s) * product 
  print("A: ", A)
  print(f"det(A) = {det}") 
  return det


def matrix_identity(n, a=1):
  I = []
  for i in range(n):
    row = [] 
    for j in range(n):
      if i == j: row.append(1*a)
      else: row.append(0) 
    I.append(row)
  return I


def matrix_scale(A, b):
  n, C = len(A), []
  for i in range(n):
    row = [] 
    for j in range(n):
      row.append(A[i][j]*b)
    C.append(row)
  return C 


def matrix_sum(A, B, subtract=0):
  C = []
  n = len(A)
  for i in range(n):
    row = []
    for j in range(n):
      if subtract: row.append(A[i][j]-B[i][j])
      else: row.append(A[i][j]+B[i][j]) 
    C.append(row)
  return C


def matrix_copy(A):
  B = []
  m, n = len(A), len(A[0])
  for i in range(m):
    row = []
    for j in range(n):
      row.append(A[i][j])
    B.append(row)
  return B


def matrix_round(A, l=4):
  B = []
  m, n = len(A), len(A[0])
  for i in range(m):
    row = []
    for j in range(n):
      row.append(round(A[i][j], l))
    B.append(row)
  return B


def matrix_inverse(A):
  assert len(A) == len(A[0])
  assert matrix_determinant(A) != 0

  n = len(A)
  A_copy = matrix_copy(A)
  A_inverse = matrix_identity(n) 
  idx = list(range(n))
  for fd in range(n): 
    fdScaler = 1.0 / A_copy[fd][fd]
    for j in range(n):
      A_copy[fd][j] *= fdScaler
      A_inverse[fd][j] *= fdScaler
    for i in idx[:fd] + idx[fd+1:]: 
      crScaler = A_copy[i][fd]
      for j in range(n):
        A_copy[i][j] = A_copy[i][j] - crScaler * A_copy[fd][j]
        A_inverse[i][j] = A_inverse[i][j] - crScaler * A_inverse[fd][j]
  return A_inverse


# newtons method
def gradient_method(f1, f2, f3, coefs, base_values, pd):
  hessian = [
    [coefs[0],        0,        0, -1], 
    [       0, coefs[1],        0, -1], 
    [       0,        0, coefs[2], -1], 
    [      -1,       -1,       -1,  0]
  ]  
  gradient = [ # lagrangian
    [f1(base_values[0])-base_values[-1]],
    [f2(base_values[1])-base_values[-1]],
    [f3(base_values[2])-base_values[-1]],
    [pd-sum(base_values[:-1])]
  ]
  print("newton's method")
  print("hessian: ", hessian)
  print("gradient (lagrangian): ", gradient) 
  hessian_inverse = matrix_round(matrix_inverse(hessian),4)
  print("hessian inverse: ", hessian_inverse)
  correction = matrix_round(matrix_product(matrix_scale(hessian_inverse,-1), gradient),4)
  print("correction: ", correction)
  x = matrix_round([
    [base_values[0]+correction[0][0]],
    [base_values[1]+correction[1][0]],
    [base_values[2]+correction[2][0]],
    [base_values[3]+correction[3][0]]
  ],4)
  print("corrected x:",x)


def dynamic_programming():
  def h1(x, prime=0):
    assert x >= 20 and x <= 100
    if x >= 20 and x <= 60:
      if prime:
        f = 8 + 2*0.024*x  
      else:
        f = 80 + 8*x + 0.024*x**2  
    elif x >= 60 and x <= 100:
      if prime:
        f = 3 + 2*0.075*x
      else:
        f = 196.4 + 3*x + 0.075*x**2
    return f
   
  def h2(x, prime=0):
    assert x >= 20 and x <= 100
    if x >= 20 and x <= 40:
      if prime:
        f = 6 + 2*0.04*x  
      else:
        f = 120 + 6*x + 0.04*x**2  
    elif x >= 40 and x <= 100:
      if prime: 
        f = 3.3333 + 2*0.08333*x 
      else: 
        f = 157.335 + 3.3333*x + 0.08333*x**2 
    return f
  
  def h3(x, prime=0):
    assert x >= 20 and x <= 100
    if x >= 20 and x <= 50:
      if prime:
        f = 4.6666 + 2*0.13333*x  
      else:
        f = 100 + 4.6666*x + 0.13333*x**2  
    elif x >= 50 and x <= 100:
      if prime:
        f = 2 + 2*0.1*x 
      else:
        f = 316.66 + 2*x + 0.1*x**2 
    return f
    
  fuel_cost = 1.5
  f1 = lambda x: fuel_cost * h1(x) 
  f2 = lambda x: fuel_cost * h2(x) 
  f3 = lambda x: fuel_cost * h3(x) 
  f1_prime = lambda x: fuel_cost * h1(x,1) 
  f2_prime = lambda x: fuel_cost * h2(x,1) 
  f3_prime = lambda x: fuel_cost * h3(x,1) 
  x = list(range(20,100))
  y1 = [f1(i) for i in x]
  y2 = [f2(i) for i in x]
  y3 = [f3(i) for i in x]
  dy1 = [f1_prime(i) for i in x]
  dy2 = [f2_prime(i) for i in x]
  dy3 = [f3_prime(i) for i in x]

  plt.title("Cost function")
  plt.plot(x, y1,color="orange", label="Unit 1") 
  plt.plot(x, y2,color="blue", label="Unit 2") 
  plt.plot(x, y3,color="green", label="Unit 3") 
  plt.xlabel("P, MW")
  plt.ylabel("F, $/h")
  plt.legend()
  plt.tight_layout()
  plt.show()

  plt.title("Incremental cost function")
  plt.plot(x, dy1,color="orange", label="Unit 1") 
  plt.plot(x, dy2,color="blue", label="Unit 2") 
  plt.plot(x, dy3,color="green", label="Unit 3") 
  plt.xlabel("P, MW")
  plt.ylabel("dF/dP, $/MWh")
  plt.legend()
  plt.tight_layout()
  plt.show()

  pd = [100, 140, 180, 220, 260]
  d = [20*i for i in range(1,14)]
  p = [20*i for i in range(1,6)]
  print("load levels D", d)
  print("power levels P", p)
  print("total demands", pd)
  print("initial data points")
  print(f"{'p1=p2=p3':>10} {'f1':>10} {'f2':>10} {'f3':>10}")
  for i in p:
    print(f"{i:>10} {round(f1(i),4):>10} {round(f2(i),4):>10} {round(f3(i),4):>10}")
  print("scheduling units 1 and 2")
  print(f"{'D':>4} {'F1(D)':>10} {' ':>50} {'f2':>10} {'p2*':>4}")
  costs = dict() 
  for i in d:
    cost = [] 
    p_star = []
    for j in p:
      if (i-j) >=20 and (i-j) <=100:
        cost.append(round(f1(i-j)+f2(j),4))
        p_star.append(j)
      else: 
        cost.append(round(float("inf"),4))
        p_star.append(j)
    costs[i]=cost
    costs_str = " ".join(list(map(str, cost)))
    if i >= 20 and i <= 100:
      print(f"{i:>4} {round(f1(i),4):>10} {costs_str:<50} {min(cost):>10} {p_star[cost.index(min(cost))]:>4}")
    else:
      print(f"{i:>4} {round(float('inf'),4):>10} {costs_str:<50} {min(cost):>10} {p_star[cost.index(min(cost))]:>4}")
  prev_costs = costs.copy() 
  print("scheduling unit 3")
  print(f"{'D':>4} {'F3(D)':>10} {'f2':>10} {' ':>50} {'f3':>10} {'p3*':>4}")
  costs = dict() 
  for i in d:
    cost = [] 
    p_star = []
    for j in p:
      if j >=20 and j <=100 and (i-j) in list(prev_costs.keys()):
        cost.append(round(min(prev_costs[(i-j)])+f3(j),4))
        p_star.append(j)
      else: 
        cost.append(round(float("inf"),4))
        p_star.append(j)
    costs[i]=cost
    costs_str = " ".join(list(map(str, cost)))
    if i >= 20 and i <= 100:
      print(f"{i:>4} {round(f3(i),4):>10} {min(prev_costs.get(i, ' ')):>10} {costs_str:<50} {min(cost):>10} {p_star[cost.index(min(cost))]:>4}")
    else:
      print(f"{i:>4} {round(float('inf'),4):>10} {min(prev_costs.get(i, ' ')):>10} {costs_str:<50} {min(cost):>10} {p_star[cost.index(min(cost))]:>4}")


def main():
  h1 = [225, 8.4, 0.0025] 
  h2 = [729, 6.3, 0.0081] 
  h3 = [400, 7.5, 0.0025] 
  # cost function
  f1 = [round(0.80*coef,4) for coef in h1]
  f2 = [round(1.02*coef,4) for coef in h2]
  f3 = [round(0.90*coef,4) for coef in h3]
  print(f"{h1}->{f1}")
  print(f"{h2}->{f2}")
  print(f"{h3}->{f3}")
  # incremental cost function
  f1_prime = [f1[1], 2*f1[2]]
  f2_prime = [f2[1], 2*f2[2]]
  f3_prime = [f3[1], 2*f3[2]]
  f1_prime_func = lambda x: (x-f1_prime[0]) / f1_prime[1]
  f2_prime_func = lambda x: (x-f2_prime[0]) / f2_prime[1]
  f3_prime_func = lambda x: (x-f3_prime[0]) / f3_prime[1]
  f1_prime_func1 = lambda x: f1_prime[0] + f1_prime[1] * x
  f2_prime_func1 = lambda x: f2_prime[0] + f2_prime[1] * x
  f3_prime_func1 = lambda x: f3_prime[0] + f3_prime[1] * x
  p1, p2, p3 = lambda_iteration(f1_prime_func, f2_prime_func, f3_prime_func)
  base_point_participation_factor(
    (f1_prime[1], f2_prime[1], f3_prime[1]), # second derivative
    (p1, p2, p3, 450)
  )
  gradient_method(
    f1_prime_func1, f2_prime_func1, f3_prime_func1,
    (f1_prime[1], f2_prime[1], f3_prime[1]), # second derivative
    (100, 300, 100, 0), # p1, p2, p3, lambda
    500
  )
  dynamic_programming()

if __name__ == "__main__":
  main()
