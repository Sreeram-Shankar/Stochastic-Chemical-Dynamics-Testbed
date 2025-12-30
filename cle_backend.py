import hashlib
import numpy as np
import multiprocessing as mp
import os
import solvers.rk as rk
import solvers.sdirk as sdirk
import solvers.linear_multistep as lms
import solvers.irk as irk
import solvers.sde as sde
from generation.multistep import get_ab_coeffs, get_am_coeffs
from generation.bdf import bdf_coeffs
import mpmath as mpmath
mpmath.dps = 200
from tqdm import tqdm

#function that derives a seed from a global seed and a label
def derive_seed(global_seed: int, label: str) -> int:
    data = f"{global_seed}:{label}".encode("utf-8")
    digest = hashlib.sha256(data).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)

#function that selects the deterministic integrator step function based on family and order
def get_deterministic_integrator_step(det_family: str, det_order: int):
    det_family = det_family.lower().strip()
    
    #gets the explicit rk step methods
    if det_family == "explicit runge-kutta":
        rk_steps = {1: rk.step_rk1, 2: rk.step_rk2, 3: rk.step_rk3, 4: rk.step_rk4, 5: rk.step_rk5, 6: rk.step_rk6, 7: rk.step_rk7}
        if det_order not in rk_steps: raise ValueError(f"Explicit Runge-Kutta order must be between 1 and 7, got {det_order}")
        return rk_steps[det_order]
    
    #gets the sdirk step methods
    elif det_family == "sdirk":
        if det_order == 2:
            gamma = 1.0 - 1.0/np.sqrt(2.0)
            A = np.array([[gamma, 0.0], [1.0 - gamma, gamma]])
            b = np.array([1.0 - gamma, gamma])
            c = np.array([gamma, 1.0])
            return lambda f, t, y, h: sdirk.step_sdirk(f, t, y, h, A, b, c)
        elif det_order == 3:
            gamma = 0.435866521508459
            A = np.array([[gamma, 0.0, 0.0], [0.2820667395, gamma, 0.0], [1.208496649, -0.644363171, gamma]])
            b = np.array([1.208496649, -0.644363171, gamma])
            c = np.array([gamma, 0.7179332605, 1.0])
            return lambda f, t, y, h: sdirk.step_sdirk(f, t, y, h, A, b, c)
        elif det_order == 4:
            gamma = 0.572816062482135
            a21 = 0.5 - gamma
            a31 = 2 * gamma
            a32 = 1 - 4 * gamma
            a41 = 2 * gamma
            a42 = 1 - 4 * gamma
            a43 = gamma
            A = np.array([[gamma, 0.0, 0.0, 0.0], [a21, gamma, 0.0, 0.0], [a31, a32, gamma, 0.0], [a41, a42, a43, gamma]])
            b = np.array([a41, a42, a43, gamma])
            c = np.array([gamma, a21 + gamma, a31 + a32 + gamma, 1.0])
            return lambda f, t, y, h: sdirk.step_sdirk(f, t, y, h, A, b, c)
        else: raise ValueError(f"SDIRK order must be between 2 and 4, got {det_order}")
    
    #gets the gauss-legendre irk step methods
    elif det_family == "gauss-legendre":
        A, b, c = irk.get_tableau("gauss", det_order)
        return lambda f, t, y, h: irk.step_collocation(f, t, y, h, A, b, c)
    
    #gets the radauIIA irk step methods
    elif det_family == "radauiia":
        A, b, c = irk.get_tableau("radau", det_order)
        return lambda f, t, y, h: irk.step_collocation(f, t, y, h, A, b, c)
    
    #gets the LobattoIIC irk step methods
    elif det_family == "lobattoiiic":
        A, b, c = irk.get_tableau("lobatto", det_order)
        return lambda f, t, y, h: irk.step_collocation(f, t, y, h, A, b, c)
    
    #creates the Adams-Bashforth step with history management
    elif det_family == "adams-bashforth":
        b = np.asarray(get_ab_coeffs(det_order), dtype=float)
        k = len(b)
        F_history = []
        last_t = None
        
        #defines the Adams-Bashforth step function
        def ab_step(f, t, y, h):
            nonlocal F_history, last_t
            #resets the history if the time goes backwards (new Monte Carlo path)
            if last_t is not None and t < last_t:
                F_history = []
                last_t = None
            
            if len(F_history) == 0:
                #bootstraps the first steps with RK4
                F_history = [f(t, y)] 
                y_current = y
                t_current = t
                for i in range(k - 1):
                    y_current = lms._rk4_step(f, t_current, y_current, h)
                    t_current += h
                    F_history.append(f(t_current, y_current))
                last_t = t_current
                return y_current
            
            #defines the Adams-Bashforth step
            acc = 0.0
            for j in range(k):
                acc += b[j] * F_history[-(j+1)]
            y_next = y + h * acc
            #updates the history by removing the oldest and adding the newest
            F_history.append(f(t + h, y_next))
            if len(F_history) > k:
                F_history.pop(0)
            last_t = t + h
            return y_next
        return ab_step
    
    #creates the Adams-Moulton step with history management
    elif det_family == "adams-moulton":
        b = np.asarray(get_am_coeffs(det_order), dtype=float)
        k = len(b)
        b0 = b[0]
        F_history = []
        last_t = None
        
        #defines the Adams-Moulton step function
        def am_step(f, t, y, h):
            nonlocal F_history, last_t
            #resets the history if the time goes backwards (new Monte Carlo path)
            if last_t is not None and t < last_t:
                F_history = []
                last_t = None
            
            if len(F_history) == 0:
                #bootstraps the first steps with RK4
                F_history = [f(t, y)] 
                y_current = y
                t_current = t
                for i in range(k - 1):
                    y_current = lms._rk4_step(f, t_current, y_current, h)
                    t_current += h
                    F_history.append(f(t_current, y_current))
                last_t = t_current
                return y_current
            
            #defines the Adams-Moulton step
            t_next = t + h
            known = sum(b[j] * F_history[-(j)] for j in range(1, k))
            def R(y_next): return y_next - y - h * (b0 * f(t_next, y_next) + known)
            def J(y_next): return np.eye(len(y)) - h * b0 * lms.finite_diff_jac(lambda z: f(t_next, z), y_next)
            
            y_guess = y
            y_next = lms.newton_solve(R, y_guess, J)
            #updates the history by removing the oldest and adding the newest
            F_history.append(f(t_next, y_next))
            if len(F_history) > k:
                F_history.pop(0)
            last_t = t_next
            return y_next
        
        return am_step
    
    #creates the BDF step with history management
    elif det_family == "bdf":
        a, beta0 = bdf_coeffs(det_order)
        a = np.asarray([float(val) for val in a], dtype=float)
        beta0 = float(beta0)
        order = det_order
        Y_history = []
        last_t = None
        
        #defines the BDF step function
        def bdf_step(f, t, y, h):
            nonlocal Y_history, last_t
            #resets the history if the time goes backwards (new Monte Carlo path)
            if last_t is not None and t < last_t:
                Y_history = []
                last_t = None
            
            if len(Y_history) == 0:
                #bootstrap with RK4 steps
                Y_history = [y] 
                y_current = y
                t_current = t
                for i in range(order - 1):
                    y_current = lms._rk4_step(f, t_current, y_current, h)
                    t_current += h
                    Y_history.append(y_current)
                last_t = t_current
                return y_current
            
            #defines the BDF step
            t_next = t + h
            known = np.zeros_like(y, dtype=float)
            for j in range(1, order + 1):
                known += a[j] * Y_history[-(j)]
            def R(y_next): return a[0] * y_next + known - beta0 * h * f(t_next, y_next)
            def J(y_next): return a[0] * np.eye(len(y)) - beta0 * h * lms.finite_diff_jac(lambda z: f(t_next, z), y_next)
            y_guess = y
            y_next = lms.newton_solve(R, y_guess, J)

            #updates the history by removing the oldest and adding the newest
            Y_history.append(y_next)
            if len(Y_history) > order:
                Y_history.pop(0)
            last_t = t_next
            return y_next
        return bdf_step
    else: raise ValueError(f"Unknown deterministic integrator family: {det_family}")

#function that selects the stochastic integrator step function based on user choice
def get_stochastic_integrator_step(stoch_integrator: str):
    #normalizes the string by replacing en-dash with hyphen and converting to lowercase
    stoch_integrator_normalized = stoch_integrator.replace("–", "-").lower().strip()
    
    #gets the euler-maruyama method
    if stoch_integrator_normalized == "euler-maruyama":
        return sde.step_euler_maruyama_with_noise
    
    #gets the milstein method
    elif stoch_integrator_normalized == "milstein":
        return sde.step_milstein_with_noise
    
    #gets the tamed euler method
    elif stoch_integrator_normalized == "tamed euler":
        return sde.step_tamed_euler_with_noise
    
    #gets the balanced / split-step euler method
    elif stoch_integrator_normalized == "balanced / split-step euler" or stoch_integrator_normalized == "balanced/split-step euler":
        return sde.step_split_step_euler_with_noise
    else: raise ValueError(f"Unknown stochastic integrator: {stoch_integrator}")

#worker function for a single SSA path (used for parallelization)
def _ssa_single_path(args_path):
    m, k1, k2, k3, k4, A, B, ssa_initial, T, N, random_seed = args_path
    
    #derives a unique seed for this path
    path_seed = derive_seed(random_seed, f"SSA_path_{m}")
    rng_ssa = np.random.default_rng(path_seed)
    t_grid = np.linspace(0, T, N + 1)
    
    #initializes the path trajectory
    Y_path = np.zeros((N + 1, 1), dtype=int)
    Y_path[0, 0] = ssa_initial
    t = 0.0
    X = int(ssa_initial)
    grid_idx = 1
    
    #performs the Gillespie Direct Method until final time T
    while t < T and grid_idx <= N:
        #computes the reaction propensities for Schlögl model
        a1 = k1 * A * X * (X - 1) if X >= 2 else 0.0
        a2 = k2 * X * (X - 1) * (X - 2) if X >= 3 else 0.0
        a3 = k3 * B
        a4 = k4 * X if X >= 1 else 0.0
        a0 = a1 + a2 + a3 + a4
        
        #stops if no reactions are possible
        if a0 <= 0:
            #fills any remaining time grid points with final state
            while grid_idx <= N:
                Y_path[grid_idx, 0] = X
                grid_idx += 1
            break
        
        #draws two uniform random numbers
        r1 = rng_ssa.random()
        r2 = rng_ssa.random()
        
        #samples the time to the next reaction
        tau = -np.log(r1) / a0 if r1 > 0 else np.inf
        t_new = t + tau
        
        #records the state at all time grid points before the next reaction
        while grid_idx <= N and t_grid[grid_idx] <= t_new:
            Y_path[grid_idx, 0] = X
            grid_idx += 1
        
        #stops if the final time has been reached
        if t_new >= T: break
        t = t_new
        
        #selects which reaction fires
        threshold = r2 * a0
        if threshold < a1: X += 1
        elif threshold < a1 + a2: X -= 1
        elif threshold < a1 + a2 + a3: X += 1 
        else: X -= 1
        
        #ensures the state stays non-negative and integer
        X = max(0, int(X))
    
    #fills any remaining time grid points with the final state
    while grid_idx <= N:
        Y_path[grid_idx, 0] = X
        grid_idx += 1
    return m, Y_path

#function that performs the SSA simulation using Gillespie Direct Method
def run_ssa_simulation(args, progress_tracker=None):
    k1, k2, k3, k4, A, B, ssa_initial, T, N, M, random_seed = args
    
    #defines the time grid
    t_grid = np.linspace(0, T, N + 1)
    Y_all = np.zeros((M, N + 1, 1), dtype=int)
    
    #initializes progress tracker if provided
    if progress_tracker is not None:
        progress_tracker.current_step = 0
        progress_tracker.total_steps = M
    
    #determines the number of worker processes
    n_cores = os.cpu_count() or 1
    n_workers = max(n_cores - 1, 1)
    
    #prepares arguments for each path
    seed_ssa = derive_seed(random_seed, "SDE")
    path_args = [(m, k1, k2, k3, k4, A, B, ssa_initial, T, N, seed_ssa) for m in range(M)]
    
    #runs paths in parallel using multiprocessing
    with mp.Pool(processes=n_workers) as pool:
        #updates progress if tracker is provided, otherwise uses tqdm
        if progress_tracker is not None:
            results = []
            for m, Y_path in pool.imap(_ssa_single_path, path_args):
                results.append((m, Y_path))
                if progress_tracker is not None:
                    progress_tracker.current_step = len(results)
        else: results = list(tqdm(pool.imap(_ssa_single_path, path_args), total=M, desc="SSA Progress", unit="path"))
    
    #collects results from all paths
    for m, Y_path in results:
        Y_all[m, :, :] = Y_path
    return t_grid, Y_all

#worker function for a single CLE path
def _cle_single_path(args_path):
    m, k1, k2, k3, k4, A, B, cle_initial, T, N, dt, det_family, det_order, stoch_integrator, op_splitting, noise_path = args_path
    
    #gets the step functions
    det_step_func = get_deterministic_integrator_step(det_family, det_order)
    stoch_step = get_stochastic_integrator_step(stoch_integrator)
    op_splitting_normalized = op_splitting.lower().strip()
    
    #defines the drift function for Schlögl model
    def drift(t, x):
        x_val = x[0] if len(x) > 0 else 0.0
        x_val = max(0.0, x_val)
        a1 = k1 * A * x_val * x_val
        a2 = k2 * x_val * x_val * x_val
        a3 = k3 * B
        a4 = k4 * x_val
        drift_val = (+1) * a1 + (-1) * a2 + (+1) * a3 + (-1) * a4
        return np.array([drift_val])
    
    #defines the diffusion function for Schlögl model
    def diffusion(t, x):
        x_val = x[0] if len(x) > 0 else 0.0
        x_val = max(0.0, x_val)
        a1 = k1 * A * x_val * x_val
        a2 = k2 * x_val * x_val * x_val
        a3 = k3 * B
        a4 = k4 * x_val
        diffusion_val = np.sqrt((+1)**2 * a1 + (-1)**2 * a2 + (+1)**2 * a3 + (-1)**2 * a4)
        return np.array([diffusion_val])
    
    #defines the derivative of diffusion function for Milstein (Schlögl model)
    def diffusion_prime(t, x):
        x_val = x[0] if len(x) > 0 else 0.0
        x_val = max(0.0, x_val)
        da1_dx = k1 * A * 2 * x_val
        da2_dx = k2 * 3 * x_val * x_val
        da4_dx = k4
        a1 = k1 * A * x_val * x_val
        a2 = k2 * x_val * x_val * x_val
        a3 = k3 * B
        a4 = k4 * x_val
        sum_inside = (+1)**2 * a1 + (-1)**2 * a2 + (+1)**2 * a3 + (-1)**2 * a4
        dsum_dx = (+1)**2 * da1_dx + (-1)**2 * da2_dx + (+1)**2 * 0 + (-1)**2 * da4_dx
        b_prime_val = (1.0 / (2.0 * np.sqrt(sum_inside))) * dsum_dx if sum_inside > 0 else 0.0
        return np.array([b_prime_val])
    
    #defines a deterministic step using the selected deterministic integrator step function
    def det_step(t, y, h):
        def f(t, y):
            return drift(t, y)
        return det_step_func(f, t, y, h)
    
    #defines a stochastic step using the selected stochastic integrator
    def stoch_step_wrapper(t, y, h, dW):
        stoch_integrator_normalized = stoch_integrator.replace("–", "-").lower().strip()
        if stoch_integrator_normalized == "milstein": return stoch_step(drift, diffusion, diffusion_prime, t, y, h, dW)
        elif stoch_integrator_normalized == "tamed euler": return stoch_step(drift, diffusion, t, y, h, dW, K=1.0)
        else: return stoch_step(drift, diffusion, t, y, h, dW)
    
    #initializes the path trajectory
    t_grid = np.linspace(0, T, N + 1)
    Y_path = np.zeros((N + 1, 1))
    Y_path[0, 0] = cle_initial
    
    y = np.array([cle_initial])
    for n in range(N):
        t = t_grid[n]
        dW = np.sqrt(dt) * noise_path[n]
        
        #applies operator splitting
        if op_splitting_normalized == "lie":
            y = det_step(t, y, dt)
            y = stoch_step_wrapper(t, y, dt, dW)
        elif op_splitting_normalized == "strang":
            y = det_step(t, y, dt / 2)
            y = stoch_step_wrapper(t + dt / 2, y, dt, dW)
            y = det_step(t + dt / 2, y, dt / 2)
        else: raise ValueError(f"Unknown operator splitting: {op_splitting}")
        y = np.maximum(y, 0.0)
        Y_path[n + 1, 0] = y[0]
    return m, Y_path

#function that runs the CLE simulation
def run_cle_simulation(args, progress_tracker=None):
    k1, k2, k3, k4, A, B, cle_initial, T, N, M, random_seed, det_family, det_order, stoch_integrator, op_splitting = args

    #defines the random simulation parameters
    dt = T / N
    seed_sde = derive_seed(random_seed, "SDE")
    rng_sde = np.random.default_rng(seed_sde)
    noise = rng_sde.normal(size=(M, N))
    
    #initializes storage
    t_grid = np.linspace(0, T, N + 1)
    Y_all = np.zeros((M, N + 1, 1))
    
    #initializes progress tracker if provided
    if progress_tracker is not None:
        progress_tracker.current_step = 0
        progress_tracker.total_steps = M
    
    #determines the number of worker processes
    n_cores = os.cpu_count() or 1
    n_workers = max(n_cores - 1, 1)
    
    #prepares arguments for each path (includes the noise slice for that path)
    path_args = [(m, k1, k2, k3, k4, A, B, cle_initial, T, N, dt, det_family, det_order, stoch_integrator, op_splitting, noise[m, :]) for m in range(M)]
    
    #runs paths in parallel using multiprocessing
    with mp.Pool(processes=n_workers) as pool:
        #updates progress if tracker is provided, otherwise uses tqdm
        if progress_tracker is not None:
            results = []
            for m, Y_path in pool.imap(_cle_single_path, path_args):
                results.append((m, Y_path))
                if progress_tracker is not None:
                    progress_tracker.current_step = len(results)
        else: results = list(tqdm(pool.imap(_cle_single_path, path_args), total=M, desc="CLE Progress", unit="path"))
    
    #collects results from all paths
    for m, Y_path in results: Y_all[m, :, :] = Y_path
    return t_grid, Y_all