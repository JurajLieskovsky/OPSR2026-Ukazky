import cvxpy as cp

a = [2000, 2500, 3000]
b = [5, 4, 3]
c = [0.02, 0.015, 0.01]
l = [200, 300, 100]
u = [1000, 1500, 800]

x = cp.Variable(3)
z = cp.Variable(3, boolean=True)

objective = cp.Minimize(a @ z + b @ x + c @ cp.square(x))
constraints = [
    cp.multiply(z, l) <= x,
    x <= cp.multiply(z, u),
    cp.sum(x) == 900,
]

problem = cp.Problem(objective, constraints)
problem.solve(cp.SCIP)

print(objective.value)
print(x.value)
