"""
Partial state knowledge (u^(y) only) with delayed input and reservoir values.

Notation follows Fleddermann, Herzog & Parlitz, Chaos 35, 053147 (2025), Sec. II D,
Table II and Fig. 3:
  N, ε, ρ, ν, β, α, Δt     classical hyperparameters
  N_I, Δt_I = i_IN Δt      number of delayed inputs and their step size
  N_R, Δt_R = i_R Δt       number of delayed reservoir states and their step size
  Delayed input (Fig. 3a), added to BOTH the reservoir input and the extended state:
      z_m = [b_in, u_m, u_{m-i_IN}, ..., u_{m-N_I i_IN}]
  Delayed reservoir (Fig. 3b), added to the extended state:
      s_m, s_{m-i_R}, ..., s_{m-N_R i_R}
  Combined (Fig. 3c):
      x_m = [b_out, s_m, s_{m-i_R}, ..., s_{m-N_R i_R}, u_m, u_{m-i_IN}, ..., u_{m-N_I i_IN}]
  y_m = W_out x_m, trained so that y_m = u_{m+1}; W_out from Eq. (2).

Method (Sec. III A, Fig. 5b):
  * Classical hyperparameters fixed at the best classical partial-state set (Level 3).
  * For every (N_I, N_R) the step sizes Δt_I, Δt_R and β are optimised.
  * Output: % deviation of the mean t_valid from the classical reservoir (N_I = N_R = 0).
  * t_valid: closed-loop prediction, E(t) of Eq. (3) on u^(y), e = 0.2, in Lyapunov
    times; mean over M reservoirs x K trajectories, uncertainty = std of reservoir means.
Like the paper, β and the delay step sizes are selected by the mean t_valid on the
evaluation trajectories. Code convention: rows are time instances / trajectories,
so M v in the paper is written v @ M.T.
"""
import numpy as np
import time

print("Level 4 ESN Pipeline: partial state (u^(y) only) with delayed values (Fig. 5b)")

# ------------------------------------------------------------------
# 1. Hyperparameters (Table II)
# ------------------------------------------------------------------
N     = 300
ε     = 0.1
ρ     = 1.0          # classical hyperparameters: best partial-state set from Level 3
ν     = 3.0
α     = 0.2
Δt_ds = 0.01
Δt    = 5 * Δt_ds

b_in  = 1.0
b_out = 1.0

# Delay hyperparameters
N_I_values  = [0, 1, 2, 3]              # number of delayed input states N_I
N_R_values  = [0, 1, 2, 3]              # number of delayed reservoir states N_R
i_IN_values = [2, 4]                    # Δt_I = i_IN Δt, best one kept for each (N_I, N_R)
i_R_values  = [2, 4]                    # Δt_R = i_R Δt
β_values    = [1e-6, 1e-5, 1e-4, 1e-3]  # β optimised for each configuration

T          = 500.0
t_trans    = 100.0
t_pred_max = 25.0       # Lyapunov times
λ_max      = 0.9056
e          = 0.2
M          = 5          # paper: 50
K          = 20         # paper: 50
seed       = 42

n_sub   = int(round(Δt / Δt_ds))
n_trans = int(round(t_trans / Δt))
n_T     = int(round(T / Δt))
n_pred  = int(round(t_pred_max / (λ_max * Δt)))

# ------------------------------------------------------------------
# 2. Lorenz-63 system, Eq. (4), RK4, rescaled to [0, 1]; only u^(y) is used
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
u_train = ((u_train - u_min) / (u_max - u_min))[:, :, 1:2]     # u^(y) only: (M, steps, 1)
u_eval  = ((u_eval - u_min) / (u_max - u_min))[:, :, 1:2]      # (K, steps, 1)

mean_u_true_sq = np.mean(u_eval[:, n_trans:, 0] ** 2)          # denominator of Eq. (3)

# ------------------------------------------------------------------
# 3. History buffer and delayed-values ESN
# ------------------------------------------------------------------
class History:
    """Stores the last L values. get(k) returns the value k steps back (k = 0: newest)."""
    def __init__(self, first, L):
        self.buf = np.repeat(np.asarray(first)[None], L, axis=0)
        self.L, self.i = L, 0

    def push(self, v):
        self.i = (self.i + 1) % self.L
        self.buf[self.i] = v

    def get(self, k):
        return self.buf[(self.i - k) % self.L]

    def copy(self):
        h = History.__new__(History)
        h.buf, h.L, h.i = self.buf.copy(), self.L, self.i
        return h


class DelayESN:
    def __init__(self, dim, seed, N_I, i_IN, N_R, i_R):
        rng = np.random.default_rng(seed)
        self.W_in = rng.uniform(-1, 1, (N, 1 + dim * (1 + N_I)))   # z_m includes delayed inputs
        W = rng.uniform(-1, 1, (N, N))
        W[rng.random((N, N)) > ε] = 0.0
        W /= np.max(np.abs(np.linalg.eigvals(W)))                  # spectral radius of unity
        self.W = W
        self.s_0 = rng.random(N)                                   # s_0 uniform in [0, 1)^N
        self.lags_I = [k * i_IN for k in range(N_I + 1)]           # 0, i_IN, ..., N_I i_IN
        self.lags_R = [k * i_R for k in range(N_R + 1)]            # 0, i_R, ..., N_R i_R
        self.n_x = 1 + N * len(self.lags_R) + dim * len(self.lags_I)   # length of x_m
        self.W_out = None

    def new_histories(self, u_first, s_first):
        return History(u_first, self.lags_I[-1] + 1), History(s_first, self.lags_R[-1] + 1)

    def z(self, u_hist):
        """z_m = [b_in, u_m, u_{m-i_IN}, ..., u_{m-N_I i_IN}]"""
        u_parts = [u_hist.get(k) for k in self.lags_I]
        bias = np.full(u_parts[0].shape[:-1] + (1,), b_in)
        return np.concatenate([bias] + u_parts, axis=-1)

    def x(self, u_hist, s_hist):
        """x_m = [b_out, s_m, s_{m-i_R}, ..., s_{m-N_R i_R}, u_m, u_{m-i_IN}, ..., u_{m-N_I i_IN}]"""
        parts = [s_hist.get(k) for k in self.lags_R] + [u_hist.get(k) for k in self.lags_I]
        bias = np.full(parts[0].shape[:-1] + (1,), b_out)
        return np.concatenate([bias] + parts, axis=-1)

    def update(self, s_prev, u_m, u_hist, s_hist):
        """Store u_m, apply Eq. (1): s_m = (1 - α) s_{m-1} + α tanh(ν W_in z_m + ρ W s_{m-1}),
        store s_m."""
        u_hist.push(u_m)
        z_m = self.z(u_hist)
        s_m = (1 - α) * s_prev + α * np.tanh(ν * (z_m @ self.W_in.T) + ρ * (s_prev @ self.W.T))
        s_hist.push(s_m)
        return s_m

    def train_matrices(self, u, chunk=2000):
        """Drive with one training trajectory u (steps, dim). Returns X X^T and Y X^T of
        Eq. (2), summed over chunks so the full matrix X is never stored."""
        s_m = self.s_0.copy()
        u_hist, s_hist = self.new_histories(u[0], s_m)
        X_XT = np.zeros((self.n_x, self.n_x))
        Y_XT = np.zeros((u.shape[1], self.n_x))
        X, Y = [], []
        for m in range(len(u) - 1):
            s_m = self.update(s_m, u[m], u_hist, s_hist)
            if m >= n_trans:                     # n_trans > largest lag, so histories are filled
                X.append(self.x(u_hist, s_hist))
                Y.append(u[m + 1])
                if len(X) == chunk or m == len(u) - 2:
                    X_c, Y_c = np.array(X).T, np.array(Y).T     # columns are x_m, y_m^true
                    X_XT += X_c @ X_c.T
                    Y_XT += Y_c @ X_c.T
                    X, Y = [], []
        return X_XT, Y_XT

    def warmup(self, u_eval):
        """Drive K evaluation trajectories with the true signal for t_trans."""
        K = u_eval.shape[0]
        s_m = np.tile(self.s_0, (K, 1))
        u_hist, s_hist = self.new_histories(u_eval[:, 0], s_m)
        for m in range(n_trans):
            s_m = self.update(s_m, u_eval[:, m], u_hist, s_hist)
        return s_m, u_hist, s_hist

    def valid_times(self, warm, u_eval):
        """Closed-loop prediction from the warmed-up state. Returns t_valid (Lyapunov times)."""
        s_m, u_hist, s_hist = warm[0].copy(), warm[1].copy(), warm[2].copy()
        K = u_eval.shape[0]
        n_valid = np.full(K, n_pred)
        alive = np.ones(K, dtype=bool)
        for j in range(n_pred):
            y_m = self.x(u_hist, s_hist) @ self.W_out.T                   # y_m = W_out x_m
            u_true = u_eval[:, n_trans + j, 0]
            E = np.abs(y_m[:, 0] - u_true) / np.sqrt(mean_u_true_sq)      # Eq. (3)
            failed = alive & ~(E < e)
            n_valid[failed] = j
            alive &= ~failed
            if not alive.any():
                break
            s_m = self.update(s_m, y_m, u_hist, s_hist)                   # closed loop: u_{m+1} = y_m
        return n_valid * Δt * λ_max

# ------------------------------------------------------------------
# 4. Evaluate one delay configuration (β optimised)
# ------------------------------------------------------------------
def evaluate(N_I, i_IN, N_R, i_R):
    t_valid_res = np.zeros((len(β_values), M))     # mean t_valid per β and reservoir
    for r in range(M):
        esn = DelayESN(1, 1000 + r, N_I, i_IN, N_R, i_R)
        X_XT, Y_XT = esn.train_matrices(u_train[r])
        warm = esn.warmup(u_eval)
        I = np.eye(esn.n_x)
        for k, β in enumerate(β_values):
            esn.W_out = np.linalg.solve(X_XT + β * I, Y_XT.T).T            # Eq. (2)
            t_valid_res[k, r] = esn.valid_times(warm, u_eval).mean()
    k_best = np.argmax(t_valid_res.mean(axis=1))
    return t_valid_res[k_best].mean(), t_valid_res[k_best].std(), β_values[k_best]

# ------------------------------------------------------------------
# 5. Grid over (N_I, N_R), step sizes Δt_I and Δt_R optimised for each
# ------------------------------------------------------------------
t0 = time.time()
best = {}
for N_I in N_I_values:
    for N_R in N_R_values:
        candidates = []
        for i_IN in (i_IN_values if N_I > 0 else [1]):
            for i_R in (i_R_values if N_R > 0 else [1]):
                t_mean, t_std, β = evaluate(N_I, i_IN, N_R, i_R)
                candidates.append((t_mean, t_std, β, i_IN, i_R))
        best[(N_I, N_R)] = max(candidates)
        t_mean, t_std, β, i_IN, i_R = best[(N_I, N_R)]
        print(f"  N_I={N_I} N_R={N_R}: t_valid = {t_mean:.2f} +/- {t_std:.2f}  "
              f"(Δt_I={i_IN}Δt, Δt_R={i_R}Δt, β={β:g})  [{time.time() - t0:.0f} s]")

# ------------------------------------------------------------------
# 6. Results: deviation from the classical reservoir, as in Fig. 5(b)
# ------------------------------------------------------------------
t_classical, std_classical = best[(0, 0)][0], best[(0, 0)][1]
print("--------------------------------------------------")
print("LEVEL 4 RESULTS SUMMARY (partial state: u^(y) only)")
print(f"Classical reservoir (N_I = N_R = 0): t_valid = {t_classical:.2f} +/- {std_classical:.2f} Lyapunov times")
print("Deviation of mean t_valid from classical reservoir [%]:")
print("N_I \\ N_R" + "".join(f"{N_R:>9d}" for N_R in N_R_values))
for N_I in reversed(N_I_values):
    row = "".join(f"{100 * (best[(N_I, N_R)][0] - t_classical) / t_classical:>+9.1f}" for N_R in N_R_values)
    print(f"{N_I:>9d}" + row)
key_best = max(best, key=lambda k: best[k][0])
t_mean, t_std, β, i_IN, i_R = best[key_best]
print(f"Best: N_I={key_best[0]}, N_R={key_best[1]}, Δt_I={i_IN}Δt, Δt_R={i_R}Δt, β={β:g} -> "
      f"t_valid = {t_mean:.2f} +/- {t_std:.2f} ({100 * (t_mean - t_classical) / t_classical:+.1f} %)")
print(f"M = {M} reservoirs x K = {K} trajectories, N = {N}, Δt = {Δt}; runtime {time.time() - t0:.0f} s")
print("--------------------------------------------------")
