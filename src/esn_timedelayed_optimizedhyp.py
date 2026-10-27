"""
Hyperparameter optimisation as described in Sec. II E of
Fleddermann, Herzog & Parlitz, Chaos 35, 053147 (2025), then the delayed-values
comparison repeated with the optimised classical hyperparameters.

Stage 1  Grid search over the classical hyperparameters Δt, ε, ρ, ν, α.
Stage 2  Nelder-Mead (scipy.optimize.minimize) over ε, ρ, ν, α, started from the
         best grid point. Reservoir seeds are fixed, as in the paper.
Stage 3  Classical hyperparameters fixed at the Stage 2 result. Grid over the delay
         hyperparameters N_I, Δt_I = i_IN Δt, N_R, Δt_R = i_R Δt.
Stage 4  Report: the chosen settings are evaluated on separate test trajectories.

β is searched at every evaluation over β_values. This only needs one extra linear
solve per β, because X X^T and Y X^T do not depend on β.

Difference from the paper: the paper chooses and reports on the same evaluation
trajectories. Here the choice uses K_val validation trajectories and the reported
numbers use K_test test trajectories, which were not used for any choice.

Notation as in the paper. Code convention: rows are time instances
or trajectories, so M v in the paper is written v @ M.T. Every array has the shape
(..., M, K, ·): M reservoirs and K trajectories are computed in one matrix product.
"""
import numpy as np
import itertools
import json
import time
from scipy.optimize import minimize

print("ESN: hyperparameter optimisation (grid search + Nelder-Mead)")

# ------------------------------------------------------------------
# 1. Settings
# ------------------------------------------------------------------
knowledge = "partial"      # "partial": input u^(y) only; "full": u^(x), u^(y), u^(z)

N     = 300                # node number (fixed here; the paper also optimises N)
Δt_ds = 0.01               # step size of the dynamical system
b_in  = 1.0
b_out = 1.0

T          = 500.0         # training length (time units)
t_trans    = 100.0         # transient time
t_pred_max = 25.0          # maximum prediction length (Lyapunov times)
λ_max      = 0.9056
e          = 0.2
M          = 5             # reservoirs                (paper: 50)
K_val      = 10            # validation trajectories   (used to choose hyperparameters)
K_test     = 20            # test trajectories         (used only for reported numbers)
seed       = 42

# Stage 1: grid of classical hyperparameters
grid = {
    "Δt": [0.03, 0.05, 0.08],      # must be multiples of Δt_ds; paper: [1, 8] · Δt_ds
    "ε":  [0.05, 0.2],
    "ρ":  [0.5, 1.0, 2.0],
    "ν":  [1.0, 3.0],
    "α":  [0.1, 0.3, 0.6],
}
β_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]      # paper range [1e-6, 1e-2]

# Stage 2: Nelder-Mead (Δt stays at the grid value because it must be a multiple of Δt_ds)
nm_params    = ["ε", "ρ", "ν", "α"]
nm_bounds    = {"ε": (1e-4, 1.0), "ρ": (0.0, 13.0), "ν": (0.01, 13.0), "α": (0.05, 1.0)}  # Table III
nm_max_evals = 60

# Stage 3: delay grid
N_I_values  = [0, 1, 2, 3]
N_R_values  = [0, 1, 2, 3]
i_IN_values = [2, 4]
i_R_values  = [2, 4]

cols, col_y = ([0, 1, 2], 1) if knowledge == "full" else ([1], 0)
dim = len(cols)

# ------------------------------------------------------------------
# 2. Lorenz-63 data, Eq. (4), RK4. Stored at Δt_ds; sub-sampled for each Δt.
# ------------------------------------------------------------------
def lorenz63(u, a=10.0, b=28.0, c=8.0 / 3.0):
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
    """n_traj trajectories, n_steps samples spaced Δt_ds, starting on the attractor."""
    u = rng.uniform([-15, -20, 5], [15, 20, 45], size=(n_traj, 3))
    for _ in range(int(round(t_settle / Δt_ds))):
        u = rk4_step(u, Δt_ds)
    out = np.empty((n_traj, n_steps, 3))
    for m in range(n_steps):
        out[:, m] = u
        u = rk4_step(u, Δt_ds)
    return out

def step_counts(Δt):
    n_sub = int(round(Δt / Δt_ds))
    assert abs(n_sub * Δt_ds - Δt) < 1e-9, "Δt must be a multiple of Δt_ds"
    n_trans = int(round(t_trans / Δt))
    n_T     = int(round(T / Δt))
    n_pred  = int(round(t_pred_max / (λ_max * Δt)))
    return n_sub, n_trans, n_T, n_pred

n_ds_train = max((c[1] + c[2]) * c[0] + 1 for c in map(step_counts, grid["Δt"]))
n_ds_eval  = max((c[1] + c[3] - 1) * c[0] + 1 for c in map(step_counts, grid["Δt"]))

t0 = time.time()
rng_data   = np.random.default_rng(seed)
base_train = generate_trajectories(M, n_ds_train, rng_data)      # one per reservoir
base_val   = generate_trajectories(K_val, n_ds_eval, rng_data)
base_test  = generate_trajectories(K_test, n_ds_eval, rng_data)

u_all = np.concatenate([b.reshape(-1, 3) for b in (base_train, base_val, base_test)])
u_min, u_max = u_all.min(axis=0), u_all.max(axis=0)
base_train, base_val, base_test = [((b - u_min) / (u_max - u_min))[:, :, cols]
                                   for b in (base_train, base_val, base_test)]
print(f"Data generated ({time.time() - t0:.0f} s)")

_data_cache = {}
def get_data(Δt):
    """Training, validation and test series sampled with step Δt."""
    if Δt not in _data_cache:
        n_sub, n_trans, n_T, n_pred = step_counts(Δt)
        d = dict(n_trans=n_trans, n_pred=n_pred,
                 u_train=base_train[:, ::n_sub][:, :n_trans + n_T + 1],
                 u_val=base_val[:, ::n_sub][:, :n_trans + n_pred],
                 u_test=base_test[:, ::n_sub][:, :n_trans + n_pred])
        for which in ("val", "test"):     # denominator of Eq. (3): <||u_true^(y)||^2>_t
            d["norm_" + which] = np.mean(d["u_" + which][:, n_trans:, col_y] ** 2)
        _data_cache[Δt] = d
    return _data_cache[Δt]

# ------------------------------------------------------------------
# 3. Reservoirs. The random numbers are drawn once per reservoir, so reservoir r keeps
#    the same entries while ε, ρ, ν, α change (fixed seeds, as in the paper's NM step).
# ------------------------------------------------------------------
raw = []
for r in range(M):
    rng = np.random.default_rng(1000 + r)
    raw.append(dict(W=rng.uniform(-1, 1, (N, N)),           # values of W
                    R=rng.random((N, N)),                   # entry kept if R < ε
                    s_0=rng.random(N),                      # s_0 uniform in [0, 1)^N
                    W_in=rng.uniform(-1, 1, (N, 1 + dim * (1 + max(N_I_values))))))
s_0 = np.stack([q["s_0"] for q in raw])[:, None, :]         # (M, 1, N)

_W_cache = {}
def W_T_stack(ε):
    """W^T of all M reservoirs for sparseness ε, each W rescaled to spectral radius 1."""
    if ε not in _W_cache:
        _W_cache.clear()
        Ws = []
        for q in raw:
            W = np.where(q["R"] < ε, q["W"], 0.0)
            r_W = np.max(np.abs(np.linalg.eigvals(W)))
            Ws.append(W / r_W if r_W > 1e-12 else W)
        _W_cache[ε] = np.stack(Ws).transpose(0, 2, 1)      # (M, N, N)
    return _W_cache[ε]


class History:
    """Last L values along axis 0. get(k) returns the value k steps back (k = 0: newest)."""
    def __init__(self, first, L):
        self.buf = np.repeat(np.asarray(first)[None], L, axis=0)
        self.L, self.i = L, 0

    def push(self, v):
        self.i = (self.i + 1) % self.L
        self.buf[self.i] = v

    def get(self, k):
        return self.buf[(self.i - k) % self.L]

    def expand(self, B):
        """Copy with a new axis of length B after the history axis (one copy per β)."""
        h = History.__new__(History)
        h.buf = np.broadcast_to(self.buf[:, None], (self.L, B) + self.buf.shape[1:]).copy()
        h.L, h.i = self.L, self.i
        return h

# ------------------------------------------------------------------
# 4. Train and evaluate: returns t_valid of every β, reservoir and trajectory
# ------------------------------------------------------------------
def evaluate(hp, delays, which, β_list):
    """hp = {Δt, ε, ρ, ν, α}; delays = (N_I, i_IN, N_R, i_R); which = "val" or "test".
    Trains M reservoirs (Eq. 1, Eq. 2), predicts iteratively, and returns t_valid in
    Lyapunov times with shape (len(β_list), M, K)."""
    Δt, ε, ρ, ν, α = hp["Δt"], hp["ε"], hp["ρ"], hp["ν"], hp["α"]
    N_I, i_IN, N_R, i_R = delays
    d = get_data(Δt)
    n_trans, n_pred = d["n_trans"], d["n_pred"]
    lags_I = [k * i_IN for k in range(N_I + 1)]
    lags_R = [k * i_R for k in range(N_R + 1)]
    n_z = 1 + dim * len(lags_I)
    n_x = 1 + N * len(lags_R) + dim * len(lags_I)
    W_T = W_T_stack(ε)
    W_in_T = np.stack([q["W_in"][:, :n_z] for q in raw]).transpose(0, 2, 1)   # (M, n_z, N)

    def update(s_prev, u_m, u_hist, s_hist):
        """Eq. (1) with z_m = [b_in, u_m, u_{m-i_IN}, ..., u_{m-N_I i_IN}]"""
        u_hist.push(u_m)
        parts = [u_hist.get(k) for k in lags_I]
        z_m = np.concatenate([np.full(parts[0].shape[:-1] + (1,), b_in)] + parts, axis=-1)
        s_m = (1 - α) * s_prev + α * np.tanh(ν * (z_m @ W_in_T) + ρ * (s_prev @ W_T))
        s_hist.push(s_m)
        return s_m

    def x(u_hist, s_hist):
        """x_m = [b_out, s_m, s_{m-i_R}, ..., u_m, u_{m-i_IN}, ...]"""
        parts = [s_hist.get(k) for k in lags_R] + [u_hist.get(k) for k in lags_I]
        return np.concatenate([np.full(parts[0].shape[:-1] + (1,), b_out)] + parts, axis=-1)

    # --- Training: drive with the training trajectories, sum X X^T and Y X^T ---
    u_tr = d["u_train"]                                   # (M, steps, dim)
    s_m = s_0.copy()                                      # (M, 1, N)
    u_hist, s_hist = History(u_tr[:, 0:1], lags_I[-1] + 1), History(s_m, lags_R[-1] + 1)
    X_XT = np.zeros((M, n_x, n_x))
    Y_XT = np.zeros((M, dim, n_x))
    X, Y = [], []
    for m in range(u_tr.shape[1] - 1):
        s_m = update(s_m, u_tr[:, m:m + 1], u_hist, s_hist)
        if m >= n_trans:
            X.append(x(u_hist, s_hist))
            Y.append(u_tr[:, m + 1:m + 2])
            if len(X) == 1000 or m == u_tr.shape[1] - 2:
                X_c, Y_c = np.concatenate(X, axis=1), np.concatenate(Y, axis=1)
                X_XT += X_c.transpose(0, 2, 1) @ X_c
                Y_XT += Y_c.transpose(0, 2, 1) @ X_c
                X, Y = [], []

    # --- Eq. (2) for every β: W_out^T = (X X^T + β I)^-1 X Y^T ---
    B = len(β_list)
    A = X_XT[None] + np.asarray(β_list)[:, None, None, None] * np.eye(n_x)
    rhs = np.broadcast_to(Y_XT.transpose(0, 2, 1)[None], (B, M, n_x, dim))
    W_out_T = np.linalg.solve(A, rhs)                     # (B, M, n_x, dim)

    # --- Transient on the evaluation trajectories (true input) ---
    u_ev, norm = d["u_" + which], d["norm_" + which]
    K = u_ev.shape[0]
    s_m = np.broadcast_to(s_0, (M, K, N)).copy()
    u_hist = History(np.broadcast_to(u_ev[None, :, 0], (M, K, dim)), lags_I[-1] + 1)
    s_hist = History(s_m, lags_R[-1] + 1)
    for m in range(n_trans):
        s_m = update(s_m, np.broadcast_to(u_ev[None, :, m], (M, K, dim)), u_hist, s_hist)

    # --- Closed loop for all β at once ---
    s_m = np.broadcast_to(s_m, (B, M, K, N)).copy()
    u_hist, s_hist = u_hist.expand(B), s_hist.expand(B)
    n_valid = np.full((B, M, K), n_pred)
    alive = np.ones((B, M, K), dtype=bool)
    for j in range(n_pred):
        y_m = x(u_hist, s_hist) @ W_out_T                 # y_m = W_out x_m, (B, M, K, dim)
        E = np.abs(y_m[..., col_y] - u_ev[:, n_trans + j, col_y]) / np.sqrt(norm)   # Eq. (3)
        failed = alive & ~(E < e)
        n_valid[failed] = j
        alive &= ~failed
        if not alive.any():
            break
        s_m = update(s_m, y_m, u_hist, s_hist)            # closed loop: u_{m+1} = y_m
    return n_valid * Δt * λ_max


def score(hp, delays):
    """Mean t_valid on the validation set for the best β. Returns (mean, std, β)."""
    t_valid = evaluate(hp, delays, "val", β_values)
    means = t_valid.mean(axis=(1, 2))
    k = int(np.argmax(means))
    return means[k], t_valid[k].mean(axis=1).std(), β_values[k]


def test_score(hp, delays, β):
    """Mean and std (of reservoir means) of t_valid on the test set."""
    t_valid = evaluate(hp, delays, "test", [β])[0]
    return t_valid.mean(), t_valid.mean(axis=1).std()


def hp_text(hp):
    return ", ".join(f"{k}={hp[k]:.4g}" for k in ("Δt", "ε", "ρ", "ν", "α"))


classical = (0, 1, 0, 1)                                  # N_I = N_R = 0
results = {"knowledge": knowledge, "N": N, "M": M, "K_val": K_val, "K_test": K_test}

# ------------------------------------------------------------------
# 5. Stage 1: grid search
# ------------------------------------------------------------------
print(f"\nStage 1: grid search ({np.prod([len(v) for v in grid.values()])} points, "
      f"β searched over {len(β_values)} values each)")
t_stage = time.time()
stage1 = []
for i, values in enumerate(itertools.product(*grid.values())):
    hp = dict(zip(grid.keys(), values))
    t_mean, t_std, β = score(hp, classical)
    stage1.append((t_mean, β, hp))
    print(f"  [{i + 1:3d}] {hp_text(hp)}  ->  t_valid = {t_mean:.2f} (β={β:g})")
stage1.sort(key=lambda r: -r[0])
best1 = stage1[0]
print(f"Best grid point: {hp_text(best1[2])}, β={best1[1]:g}, t_valid(val) = {best1[0]:.2f}  "
      f"[{time.time() - t_stage:.0f} s]")
results["stage1_best"] = dict(hp=best1[2], β=best1[1], t_valid_val=best1[0])

# ------------------------------------------------------------------
# 6. Stage 2: Nelder-Mead from the best grid point
# ------------------------------------------------------------------
print(f"\nStage 2: Nelder-Mead over {nm_params} (at most {nm_max_evals} evaluations)")
t_stage = time.time()
Δt_best = best1[2]["Δt"]
stage2 = [best1]

def objective(θ):
    hp = dict(zip(nm_params, (float(v) for v in θ)))
    hp["Δt"] = Δt_best
    t_mean, t_std, β = score(hp, classical)
    stage2.append((t_mean, β, hp))
    print(f"  [{len(stage2) - 1:3d}] {hp_text(hp)}  ->  t_valid = {t_mean:.2f} (β={β:g})")
    return -t_mean                                        # minimize -t_valid

θ0 = np.array([best1[2][p] for p in nm_params])
simplex = [θ0]                                            # start simplex: each parameter
for k, p in enumerate(nm_params):                         # moved by 30 % in turn
    θ = θ0.copy()
    lo, hi = nm_bounds[p]
    θ[k] = θ0[k] * 1.3 if θ0[k] * 1.3 <= hi else θ0[k] * 0.7
    θ[k] = max(θ[k], lo)
    simplex.append(θ)

minimize(objective, θ0, method="Nelder-Mead",
         bounds=[nm_bounds[p] for p in nm_params],
         options=dict(maxfev=nm_max_evals, initial_simplex=np.array(simplex),
                      xatol=1e-3, fatol=0.01))
best2 = max(stage2, key=lambda r: r[0])                   # best point evaluated
hp_classical, β_classical = best2[2], best2[1]
print(f"Best after Nelder-Mead: {hp_text(hp_classical)}, β={β_classical:g}, "
      f"t_valid(val) = {best2[0]:.2f}  [{time.time() - t_stage:.0f} s]")
results["stage2_best"] = dict(hp=hp_classical, β=β_classical, t_valid_val=best2[0])

# ------------------------------------------------------------------
# 7. Stage 3: delay grid with the classical hyperparameters fixed
# ------------------------------------------------------------------
print(f"\nStage 3: delayed values, classical hyperparameters fixed at {hp_text(hp_classical)}")
t_stage = time.time()
stage3 = {}
for N_I in N_I_values:
    for N_R in N_R_values:
        candidates = []
        for i_IN in (i_IN_values if N_I > 0 else [1]):
            for i_R in (i_R_values if N_R > 0 else [1]):
                t_mean, t_std, β = score(hp_classical, (N_I, i_IN, N_R, i_R))
                candidates.append((t_mean, β, i_IN, i_R))
        stage3[(N_I, N_R)] = max(candidates)
        t_mean, β, i_IN, i_R = stage3[(N_I, N_R)]
        print(f"  N_I={N_I} N_R={N_R}: t_valid(val) = {t_mean:.2f}  "
              f"(Δt_I={i_IN}Δt, Δt_R={i_R}Δt, β={β:g})  [{time.time() - t_stage:.0f} s]")

# ------------------------------------------------------------------
# 8. Stage 4: evaluate the chosen settings on the test trajectories
# ------------------------------------------------------------------
print("\nStage 4: test trajectories")
t_grid, s_grid = test_score(best1[2], classical, best1[1])
test3 = {}
for (N_I, N_R), (_, β, i_IN, i_R) in stage3.items():
    test3[(N_I, N_R)] = test_score(hp_classical, (N_I, i_IN, N_R, i_R), β)
t_cl, s_cl = test3[(0, 0)]

print("--------------------------------------------------")
print(f"RESULTS SUMMARY ({knowledge} state, N = {N}, M = {M}, K_test = {K_test})")
print(f"Classical, grid search only : t_valid = {t_grid:.2f} +/- {s_grid:.2f}   ({hp_text(best1[2])}, β={best1[1]:g})")
print(f"Classical, grid + Nelder-Mead: t_valid = {t_cl:.2f} +/- {s_cl:.2f}   ({hp_text(hp_classical)}, β={β_classical:g})")
print("Deviation of mean t_valid from the optimised classical reservoir [%] (test set):")
print("N_I \\ N_R" + "".join(f"{N_R:>9d}" for N_R in N_R_values))
for N_I in reversed(N_I_values):
    print(f"{N_I:>9d}" + "".join(f"{100 * (test3[(N_I, N_R)][0] - t_cl) / t_cl:>+9.1f}"
                                 for N_R in N_R_values))
key_best = max(stage3, key=lambda k: stage3[k][0])        # chosen by validation score
_, β, i_IN, i_R = stage3[key_best]
t_b, s_b = test3[key_best]
print(f"Best delayed values: N_I={key_best[0]}, N_R={key_best[1]}, Δt_I={i_IN}Δt, Δt_R={i_R}Δt, "
      f"β={β:g} -> t_valid = {t_b:.2f} +/- {s_b:.2f} ({100 * (t_b - t_cl) / t_cl:+.1f} %)")
print("(t_valid in Lyapunov times; uncertainty = std of the M reservoir means)")
print(f"Total runtime: {time.time() - t0:.0f} s")
print("--------------------------------------------------")

results["test"] = dict(classical_grid=[t_grid, s_grid], classical_nm=[t_cl, s_cl],
                       delays={f"N_I={a},N_R={b}": dict(t_valid=v[0], std=v[1], β=stage3[(a, b)][1],
                                                        i_IN=stage3[(a, b)][2], i_R=stage3[(a, b)][3])
                               for (a, b), v in test3.items()})
with open("level5_results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("Saved level5_results.json")
