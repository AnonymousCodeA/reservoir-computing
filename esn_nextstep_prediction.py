"""
Classical echo state network (ESN), one-step-ahead prediction of
Lorenz-63 with FULL state knowledge.

Notation follows Fleddermann, Herzog & Parlitz, Chaos 35, 053147 (2025), Sec. II A-C
and Table II:
  N, ε, ρ, ν, β, α, Δt            classical hyperparameters
  u_m = u(t_m)                    driving input at t_m = t_0 + m Δt
  z_m = [b_in, u_m]               reservoir input
  s_m                             reservoir state, Eq. (1)
  x_m = [b_out, s_m, u_m]         extended state vector
  y_m = W_out x_m                 prediction of u_{m+1}
  X, Y                            extended state matrix and target matrix, Eq. (2)
  E(t), e                         normalised error and error tolerance, Eq. (3)

Code convention: time instances / trajectories are stored as rows, so a product
written M v in the paper appears as v @ M.T in the code.

One-step-ahead (open-loop) prediction is NOT the task used for the paper's
results (iterative prediction, see Level 3). It is the first check that the ESN
is built and trained correctly. E of Eq. (3) is applied to one-step predictions.
"""
import numpy as np
import time

print("Level 2 ESN Pipeline: one-step-ahead prediction, Lorenz-63, full state")

# ------------------------------------------------------------------
# 1. Hyperparameters 
# ------------------------------------------------------------------
N     = 300          # node number
ε     = 0.1          # sparseness: only a fraction ε of the entries of W is kept
ρ     = 0.95         # spectral radius
ν     = 0.5          # input scaling
β     = 1e-4         # regularization constant
α     = 0.4          # leaking rate
Δt_ds = 0.01         # step size of the dynamical system
Δt    = 5 * Δt_ds    # step size (paper: Δt in [1, 8] · Δt_ds)

b_in  = 1.0          # constant input bias
b_out = 1.0          # constant output bias

# Experimental setup 
T       = 500.0      # training length (time units)
t_trans = 100.0      # transient time: reservoir driven, nothing recorded or evaluated
t_eval  = 20.0       # length of each evaluation window (time units)
M       = 5          # number of random reservoirs   (paper: 50)
K       = 20         # evaluation trajectories       (paper: 50)
seed    = 42

n_sub   = int(round(Δt / Δt_ds))     # integration steps per step Δt
n_trans = int(round(t_trans / Δt))
n_T     = int(round(T / Δt))
n_eval  = int(round(t_eval / Δt))

# ------------------------------------------------------------------
# 2. Lorenz-63 system, integrated with RK4
# ------------------------------------------------------------------
def lorenz63(u, a=10.0, b=28.0, c=8.0 / 3.0):
    """Eq. (4). Columns of u are u^(x), u^(y), u^(z)."""
    u_x, u_y, u_z = u[:, 0], u[:, 1], u[:, 2]
    du_x = a * (u_y - u_x)
    du_y = u_x * (b - u_z) - u_y
    du_z = u_x * u_y - c * u_z
    return np.stack((du_x, du_y, du_z), axis=1)

def rk4_step(u, h):
    k1 = lorenz63(u)
    k2 = lorenz63(u + 0.5 * h * k1)
    k3 = lorenz63(u + 0.5 * h * k2)
    k4 = lorenz63(u + h * k3)
    return u + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def generate_trajectories(n_traj, n_steps, rng, t_settle=50.0):
    """n_traj trajectories with n_steps samples spaced Δt. Random initial points are
    integrated for t_settle first so every trajectory starts on the attractor.
    Returns u with shape (n_traj, n_steps, 3)."""
    u = rng.uniform([-15, -20, 5], [15, 20, 45], size=(n_traj, 3))
    for _ in range(int(round(t_settle / Δt_ds))):
        u = rk4_step(u, Δt_ds)
    out = np.empty((n_traj, n_steps, 3))
    for m in range(n_steps):
        out[:, m] = u
        for _ in range(n_sub):
            u = rk4_step(u, Δt_ds)
    return out

rng_data = np.random.default_rng(seed)
u_train = generate_trajectories(M, n_trans + n_T + 1, rng_data)    # one per reservoir
u_eval  = generate_trajectories(K, n_trans + n_eval + 1, rng_data)

# Rescale every variable to [0, 1] with min/max over all training and testing trajectories
u_all = np.concatenate((u_train.reshape(-1, 3), u_eval.reshape(-1, 3)))
u_min, u_max = u_all.min(axis=0), u_all.max(axis=0)
u_train = (u_train - u_min) / (u_max - u_min)
u_eval  = (u_eval - u_min) / (u_max - u_min)

# ------------------------------------------------------------------
# 3. Echo state network 
# ------------------------------------------------------------------
class ESN:
    def __init__(self, dim, seed):
        rng = np.random.default_rng(seed)
        self.W_in = rng.uniform(-1, 1, (N, 1 + dim))     # couples z_m = [b_in, u_m] to the nodes
        W = rng.uniform(-1, 1, (N, N))
        W[rng.random((N, N)) > ε] = 0.0                  # keep only a fraction ε
        W /= np.max(np.abs(np.linalg.eigvals(W)))        # spectral radius of unity
        self.W = W
        self.s_0 = rng.random(N)                         # s_0 uniform in [0, 1)^N
        self.W_out = None

    @staticmethod
    def z(u_m):
        """z_m = [b_in, u_m]"""
        return np.concatenate((np.full(u_m.shape[:-1] + (1,), b_in), u_m), axis=-1)

    @staticmethod
    def x(s_m, u_m):
        """x_m = [b_out, s_m, u_m]"""
        return np.concatenate((np.full(s_m.shape[:-1] + (1,), b_out), s_m, u_m), axis=-1)

    def update(self, s_prev, u_m):
        """Eq. (1): s_m = (1 - α) s_{m-1} + α tanh(ν W_in z_m + ρ W s_{m-1})
        Works for one trajectory (s:(N,), u:(dim,)) or a batch (s:(K,N), u:(K,dim))."""
        z_m = self.z(u_m)
        return (1 - α) * s_prev + α * np.tanh(ν * (z_m @ self.W_in.T) + ρ * (s_prev @ self.W.T))

    def train(self, u):
        """u: (n_steps, dim), one training trajectory. Target of x_m is y_m = u_{m+1}."""
        s_m = self.s_0.copy()
        X, Y = [], []
        for m in range(len(u) - 1):
            s_m = self.update(s_m, u[m])
            if m >= n_trans:                             # discard transient
                X.append(self.x(s_m, u[m]))
                Y.append(u[m + 1])
        X, Y = np.array(X).T, np.array(Y).T              # columns are x_m and y_m^true
        # Eq. (2): W_out = Y X^T (X X^T + β I)^-1, solved without forming the inverse
        I = np.eye(X.shape[0])
        self.W_out = np.linalg.solve(X @ X.T + β * I, (Y @ X.T).T).T

# ------------------------------------------------------------------
# 4. Train M reservoirs and evaluate one-step predictions on K trajectories
# ------------------------------------------------------------------
# Denominator of Eq. (3): temporal mean <||u_true||^2>_t over all evaluation data
mean_u_true_sq = np.mean(np.sum(u_eval[:, n_trans + 1:] ** 2, axis=2))

mean_E   = np.zeros(M)          # mean normalised one-step error per reservoir
mse_dims = np.zeros((M, 3))
t0 = time.time()

for r in range(M):
    esn = ESN(dim=3, seed=1000 + r)
    esn.train(u_train[r])

    # Drive all K evaluation trajectories in parallel with the true signal (open loop)
    s_m = np.tile(esn.s_0, (K, 1))
    y = np.empty((K, n_eval, 3))
    for m in range(n_trans + n_eval):
        s_m = esn.update(s_m, u_eval[:, m])
        if m >= n_trans:
            y[:, m - n_trans] = esn.x(s_m, u_eval[:, m]) @ esn.W_out.T    # y_m = W_out x_m
    u_true = u_eval[:, n_trans + 1: n_trans + n_eval + 1]

    E = np.linalg.norm(y - u_true, axis=2) / np.sqrt(mean_u_true_sq)    # Eq. (3), shape (K, n_eval)
    mean_E[r] = E.mean()
    mse_dims[r] = np.mean((y - u_true) ** 2, axis=(0, 1))
    print(f"  reservoir {r + 1}/{M}: mean one-step E = {mean_E[r]:.2e}")

# ------------------------------------------------------------------
# 5. Results
# ------------------------------------------------------------------
print("--------------------------------------------------")
print("LEVEL 2 RESULTS SUMMARY (one-step-ahead, full state):")
print(f"M = {M} reservoirs, K = {K} evaluation trajectories, Δt = {Δt}")
print(f"u^(x) MSE: {mse_dims[:, 0].mean():.3e}")
print(f"u^(y) MSE: {mse_dims[:, 1].mean():.3e}")
print(f"u^(z) MSE: {mse_dims[:, 2].mean():.3e}")
print(f"Mean normalised one-step error E (Eq. 3): {mean_E.mean():.3e} +/- {mean_E.std():.1e}")
print(f"(uncertainty = std over the M reservoir means; runtime {time.time() - t0:.1f} s)")
print("--------------------------------------------------")
