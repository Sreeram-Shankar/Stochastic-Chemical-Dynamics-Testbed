import numpy as np
import mpmath as mp
mp.dps = 200
from generation.gauss_legendre import build_gauss_legendre_irk
from generation.radau import build_radau_irk
from generation.lobatto import build_lobatto_IIIC_irk

#defines the finite difference Jacobian
def finite_diff_jac(fun, x, eps=1e-8):
    n = len(x)
    f0 = fun(x)
    J = np.zeros((n, n))
    for j in range(n):
        dx = np.zeros(n)
        step = eps * max(1.0, abs(x[j]))
        dx[j] = step
        f1 = fun(x + dx)
        J[:, j] = (f1 - f0) / step
    return J

#solves the nonlinear system of equations
def newton_solve(residual, y0, jac=None, tol=1e-10, max_iter=12):
    y = y0.copy()
    for _ in range(max_iter):
        r = residual(y)
        if np.linalg.norm(r) < tol:
            return y
        J = jac(y) if jac else finite_diff_jac(residual, y)
        dy = np.linalg.solve(J, -r)
        y += dy
        if np.linalg.norm(dy) < tol:
            break
    return y

#loads the Butcher tableau from the generators
def get_tableau(family, s):
    family = family.lower()
    if family == "gauss":
        A, b, c = build_gauss_legendre_irk(s)
    elif family == "radau":
        A, b, c = build_radau_irk(s)
    elif family == "lobatto":
        A, b, c = build_lobatto_IIIC_irk(s)
    else: raise ValueError(f"Unknown family '{family}', must be 'gauss', 'radau', or 'lobatto'.")

    #converts the tableau to numpy arrays
    A = np.array([[float(A[i][j]) for j in range(s)] for i in range(s)])
    b = np.array([float(b[i]) for i in range(s)])
    c = np.array([float(c[i]) for i in range(s)])
    return A, b, c


#defines the IRK collocation step
def step_collocation(f, t, y, h, A, b, c, jac=None, tol=1e-10, max_iter=12, fd_eps=1e-8):
    s = len(b)
    n = len(y)
    Y = np.tile(y, (s, 1))
    t_nodes = t + c * h

    #builds the residual
    def residual(z_flat):
        Z = z_flat.reshape(s, n)
        R = np.zeros_like(Z)
        for i in range(s):
            acc = np.zeros(n)
            for j in range(s):
                acc += A[i, j] * f(t_nodes[j], Z[j])
            R[i] = Z[i] - y - h * acc
        return R.ravel()

    #builds the Jacobian
    def jacobian(z_flat):
        Z = z_flat.reshape(s, n)
        J_full = np.zeros((s * n, s * n))
        for j in range(s):
            Jf_j = jac(t_nodes[j], Z[j]) if jac else finite_diff_jac(lambda z: f(t_nodes[j], z), Z[j], eps=fd_eps)
            for i in range(s):
                block = -h * A[i, j] * Jf_j
                if i == j:
                    block = block + np.eye(n)
                row = slice(i * n, (i + 1) * n)
                col = slice(j * n, (j + 1) * n)
                J_full[row, col] = block
        return J_full

    z0 = Y.ravel()
    z_star = newton_solve(residual, z0, jac=jacobian, tol=tol, max_iter=max_iter)
    Y = z_star.reshape(s, n)
    K = np.zeros((s, n))
    for i in range(s):
        K[i] = f(t_nodes[i], Y[i])
    y_next = y + h * np.sum(b[:, None] * K, axis=0)
    return y_next


#main solver for any collocation method
def solve_collocation(f, t_span, y0, h, family="gauss", s=3, jac=None, tol=1e-10, max_iter=12, fd_eps=1e-8):
    A, b, c = get_tableau(family, s)
    t0, tf = t_span
    N = int(np.ceil((tf - t0)/h))
    t_grid = np.linspace(t0, tf, N+1)
    Y = np.zeros((N+1, len(y0)))
    Y[0] = y0
    for n in range(N):
        Y[n+1] = step_collocation(f, t_grid[n], Y[n], h, A, b, c, jac=jac, tol=tol, max_iter=max_iter, fd_eps=fd_eps)
    return t_grid, Y