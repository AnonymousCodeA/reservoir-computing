"""
 Iterative (closed-loop) prediction of Lorenz-63 and valid time t_valid.

Notation follows Fleddermann, Herzog & Parlitz, Chaos 35, 053147 (2025):
  N, ε, ρ, ν, β, α, Δt            classical hyperparameters (Table II)
  u_m, z_m = [b_in, u_m], s_m     input, reservoir input, reservoir state (Eq. 1)
  x_m = [b_out, s_m, u_m]         extended state vector
  y_m = W_out x_m                 one-step prediction, trained so that y_m = u_{m+1}
  E(t), e, t_valid                normalised error, tolerance, valid time (Eq. 3)
  λ_max                           largest Lyapunov exponent (0.9056 for Lorenz-63)

Method:
  * Sec. II B: y_m is fed back as the next input u_{m+1} (closed loop).
    t_valid = time until E(t) first reaches e = 0.2.
  * Sec. II C: E(t) is computed on u^(y) only, so full-state and partial-state
    runs are compared on the same quantity. t_valid is given in Lyapunov times
    (t · λ_max).
  * Sec. III A / Fig. 4(a),(b): four cases
        full state (u^(x), u^(y), u^(z))  vs  partial state (u^(y) only)
        with memory  vs  without memory (ρ = 0, α = 1)
  * Mean t_valid over M reservoirs x K evaluation trajectories; uncertainty =
    std of the M per-reservoir means.

Code convention: time instances / trajectories are rows, so M v in the paper is
written v @ M.T. The classical hyperparameters come from a small grid search,
not the paper's grid search + Nelder-Mead, so values are below Fig. 4.
"""
import numpy as np
import time

print("Level 3 ESN Pipeline: iterative prediction and valid time (Fig. 4 comparison)")

# ------------------------------------------------------------------
# 1. Hyperparameters (Table II)
# ------------------------------------------------------------------
N     = 300          # node number
ε     = 0.1          # sparseness
Δt_ds = 0.01         # step size of the dynamical system
Δt    = 5 * Δt_ds    # step size

b_in  = 1.0
b_out = 1.0

# Hyperparameters per case. Without memory: ρ = 0 and α = 1 (Sec. III A); ν and β are
# chosen for each case separately, since the paper optimises each case individually.
# Values from a small grid search (N = 300, Δt = 0.05), β kept in the paper's range
# [1e-6, 1e-2] (Table III).
cases = {
    ("full",    "memory"):    dict(ρ=0.5, ν=1.5, α=0.5, β=1e-6),
    ("full",    "no memory"): dict(ρ=0.0, ν=1.5, α=1.0, β=1e-6),
    ("partial", "memory"):    dict(ρ=1.0, ν=3.0, α=0.2, β=1e-6),
    ("partial", "no memory"): dict(ρ=0.0, ν=1.5, α=1.0, β=1e-6),
}

T          = 500.0     # training length (time units)
t_trans    = 100.0     # transient time before training and before every prediction
t_pred_max = 25.0      # maximum prediction length, in Lyapunov times
λ_max      = 0.9056    # largest Lyapunov exponent of Lorenz-63 (Sec. II C)
e          = 0.2       # error tolerance of Eq. (3)
M          = 5         # random reservoirs        (paper: 50)
K          = 20        # evaluation trajectories  (paper: 50)
seed       = 42

n_sub   = int(round(Δt / Δt_ds))
n_trans = int(round(t_trans / Δt))
n_T     = int(round(T / Δt))
n_pred  = int(round(t_pred_max / (λ_max * Δt)))

# ------------------------------------------------------------------
# 2. Lorenz-63 system, Eq. (4), RK4, rescaled to [0, 1]
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
u_train = generate_trajectories(M, n_trans + n_T + 1, rng_data)
u_eval  = generate_trajectories(K, n_trans + n_pred, rng_data)

u_all = np.concatenate((u_train.reshape(-1, 3), u_eval.reshape(-1, 3)))
u_min, u_max = u_all.min(axis=0), u_all.max(axis=0)
u_train = (u_train - u_min) / (u_max - u_min)
u_eval  = (u_eval - u_min) / (u_max - u_min)

# Denominator of Eq. (3): temporal mean <||u_true||^2>_t of u^(y)
mean_u_true_sq = np.mean(u_eval[:, n_trans:, 1] ** 2)

# ------------------------------------------------------------------
# 3. Echo state network (Sec. II A)
# ------------------------------------------------------------------
class ESN:
    def __init__(self, dim, seed, ρ, ν, α, β):
        rng = np.random.default_rng(seed)
        self.W_in = rng.uniform(-1, 1, (N, 1 + dim))
        W = rng.uniform(-1, 1, (N, N))
        W[rng.random((N, N)) > ε] = 0.0
        W /= np.max(np.abs(np.linalg.eigvals(W)))        # spectral radius of unity
        self.W = W
        self.s_0 = rng.random(N)                         # s_0 uniform in [0, 1)^N
        self.ρ, self.ν, self.α, self.β = ρ, ν, α, β
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
        """Eq. (1): s_m = (1 - α) s_{m-1} + α tanh(ν W_in z_m + ρ W s_{m-1})"""
        ρ, ν, α = self.ρ, self.ν, self.α
        z_m = self.z(u_m)
        return (1 - α) * s_prev + α * np.tanh(ν * (z_m @ self.W_in.T) + ρ * (s_prev @ self.W.T))

    def train(self, u):
        s_m = self.s_0.copy()
        X, Y = [], []
        for m in range(len(u) - 1):
            s_m = self.update(s_m, u[m])
            if m >= n_trans:
                X.append(self.x(s_m, u[m]))
                Y.append(u[m + 1])
        X, Y = np.array(X).T, np.array(Y).T
        I = np.eye(X.shape[0])
        self.W_out = np.linalg.solve(X @ X.T + self.β * I, (Y @ X.T).T).T   # Eq. (2)

    def valid_times(self, u_eval, col_y):
        """Closed-loop prediction on K trajectories at once.
        u_eval: (K, n_trans + n_pred, dim). col_y: column of u^(y) in the input.
        Returns t_valid of each trajectory in Lyapunov times."""
        K = u_eval.shape[0]
        s_m = np.tile(self.s_0, (K, 1))
        for m in range(n_trans):                         # transient: driven by the true signal
            s_m = self.update(s_m, u_eval[:, m])
        u_m = u_eval[:, n_trans - 1]
        n_valid = np.full(K, n_pred)
        alive = np.ones(K, dtype=bool)
        for j in range(n_pred):
            y_m = self.x(s_m, u_m) @ self.W_out.T                         # y_m = W_out x_m
            u_true = u_eval[:, n_trans + j, col_y]
            E = np.abs(y_m[:, col_y] - u_true) / np.sqrt(mean_u_true_sq)  # Eq. (3)
            failed = alive & ~(E < e)                    # "~(E < e)" also catches NaN
            n_valid[failed] = j
            alive &= ~failed
            if not alive.any():
                break
            u_m = y_m                                    # closed loop: u_{m+1} = y_m
            s_m = self.update(s_m, u_m)
        return n_valid * Δt * λ_max                      # t_valid in Lyapunov times

# ------------------------------------------------------------------
# 4. Run the four cases of Fig. 4(a),(b)
# ------------------------------------------------------------------
results = {}
t0 = time.time()
for (knowledge, memory), hp in cases.items():
    if knowledge == "full":
        cols, col_y = [0, 1, 2], 1                       # u^(x), u^(y), u^(z)
    else:
        cols, col_y = [1], 0                             # u^(y) only
    t_valid_res = np.zeros(M)                            # mean t_valid per reservoir
    for r in range(M):
        esn = ESN(dim=len(cols), seed=1000 + r, **hp)
        esn.train(u_train[r][:, cols])
        t_valid_res[r] = esn.valid_times(u_eval[:, :, cols], col_y).mean()
    results[(knowledge, memory)] = t_valid_res
    print(f"  {knowledge:7s} state, {memory:9s}: t_valid = {t_valid_res.mean():.2f} "
          f"+/- {t_valid_res.std():.2f} Lyapunov times")

# ------------------------------------------------------------------
# 5. Results
# ------------------------------------------------------------------
print("--------------------------------------------------")
print("LEVEL 3 RESULTS SUMMARY (iterative prediction, E(t) on u^(y)):")
print(f"M = {M} reservoirs x K = {K} trajectories, N = {N}, Δt = {Δt}")
print(f"{'':16s}{'with memory':>18s}{'no memory':>18s}")
for knowledge in ("full", "partial"):
    a, b = results[(knowledge, "memory")], results[(knowledge, "no memory")]
    print(f"{knowledge + ' state':16s}{a.mean():>10.2f} +/- {a.std():<4.2f}{b.mean():>10.2f} +/- {b.std():<4.2f}")
print("(t_valid in Lyapunov times 1/λ_max; uncertainty = std of reservoir means)")
print(f"Runtime: {time.time() - t0:.1f} s")
print("--------------------------------------------------")
