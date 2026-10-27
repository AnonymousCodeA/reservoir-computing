"""
Level 1 - Classical echo state network (ESN), prediction of a sine wave (toy problem).

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
  t_valid                         valid time: max t such that E(t) < e

Code convention: vectors are columns, as in the paper, so a product written M v
in the paper appears as M @ v in the code, and X, Y store x_m and y_m^true as columns.

Two tests are run on the held-out part of the signal:
  (a) one-step-ahead (open loop): the reservoir is driven by the true u_m
  (b) iterative (closed loop): y_m is fed back as the next input u_{m+1}
"""
import numpy as np

print("Level 1 ESN Pipeline: sine wave, one-step-ahead and iterative prediction")

# ------------------------------------------------------------------
# 1. Hyperparameters
# ------------------------------------------------------------------
N     = 100          # node number
ε     = 0.1          # sparseness: only a fraction ε of the entries of W is kept
ρ     = 0.95         # spectral radius
ν     = 1.0          # input scaling
α     = 0.3          # leaking rate
β     = 1e-6         # regularization constant
Δt    = 0.1          # step size
e     = 0.2          # error tolerance for the valid time

b_in  = 1.0          # constant input bias
b_out = 1.0          # constant output bias

# Experimental setup (lengths in steps of Δt)
n_trans = 100        # transient: reservoir driven, nothing recorded or evaluated
n_T     = 2000       # training length
n_test  = 500        # test length
seed    = 42

rng = np.random.default_rng(seed)

# ------------------------------------------------------------------
# 2. Data: sine wave, rescaled to [0, 1]
# ------------------------------------------------------------------
n_total = n_trans + n_T + n_test + 1
u   = 0.5 * (1 + np.sin(Δt * np.arange(n_total)))    # u_m, shape (n_total,)
dim = 1                                               # dimension of u

# ------------------------------------------------------------------
# 3. Echo state network
# ------------------------------------------------------------------
W_in = rng.uniform(-1, 1, (N, 1 + dim))              # couples z_m = [b_in, u_m] to the nodes
W = rng.uniform(-1, 1, (N, N))
W[rng.random((N, N)) > ε] = 0.0                      # keep only a fraction ε
W /= np.max(np.abs(np.linalg.eigvals(W)))            # spectral radius of unity (ρ enters in Eq. 1)

def z(u_m):
    """z_m = [b_in, u_m]"""
    return np.concatenate(([b_in], np.atleast_1d(u_m)))

def x(s_m, u_m):
    """x_m = [b_out, s_m, u_m]"""
    return np.concatenate(([b_out], s_m, np.atleast_1d(u_m)))

def update(s_prev, u_m):
    """Eq. (1): s_m = (1 - α) s_{m-1} + α tanh(ν W_in z_m + ρ W s_{m-1})"""
    z_m = z(u_m)
    return (1 - α) * s_prev + α * np.tanh(ν * W_in @ z_m + ρ * W @ s_prev)

# ------------------------------------------------------------------
# 4. Training: drive the reservoir with the true u, record x_m and target u_{m+1}
# ------------------------------------------------------------------
s_0 = rng.random(N)                                  # s_0 uniform in [0, 1)^N
s_m = s_0.copy()
X = np.zeros((1 + N + dim, n_T))                     # columns are x_m
Y = np.zeros((dim, n_T))                             # columns are y_m^true = u_{m+1}
for m in range(n_trans + n_T):
    s_m = update(s_m, u[m])
    if m >= n_trans:                                 # discard transient
        k = m - n_trans
        X[:, k] = x(s_m, u[m])
        Y[:, k] = u[m + 1]

# Eq. (2): W_out = Y X^T (X X^T + β I)^-1, solved without forming the inverse
I = np.eye(X.shape[0])
W_out = np.linalg.solve(X @ X.T + β * I, X @ Y.T).T

mse_train = np.mean((W_out @ X - Y) ** 2)

# ------------------------------------------------------------------
# 5. Test (a): one-step-ahead prediction, reservoir driven by the true signal
# ------------------------------------------------------------------
m_0 = n_trans + n_T                                  # index of the first test input
s_a = s_m.copy()                                     # start from the last training state
y_a    = np.zeros((dim, n_test))
u_true = np.zeros((dim, n_test))
for k in range(n_test):
    s_a = update(s_a, u[m_0 + k])
    y_a[:, k]    = W_out @ x(s_a, u[m_0 + k])        # y_m = W_out x_m
    u_true[:, k] = u[m_0 + k + 1]
mse_onestep = np.mean((y_a - u_true) ** 2)
mse_persist = np.mean((u[m_0:m_0 + n_test] - u_true[0]) ** 2)   # baseline y_m = u_m

# ------------------------------------------------------------------
# 6. Test (b): iterative (closed-loop) prediction, y_m fed back as u_{m+1}
# ------------------------------------------------------------------
s_b = s_m.copy()
y_b = np.zeros((dim, n_test))
u_m = u[m_0]                                         # only the first input is true data
for k in range(n_test):
    s_b = update(s_b, u_m)
    y_b[:, k] = W_out @ x(s_b, u_m)
    u_m = y_b[:, k]

# Eq. (3): E(t) = ||u(t) - u_true(t)|| / <||u_true||^2>_t^(1/2)
mean_u_true_sq = np.mean(np.sum(u_true ** 2, axis=0))
E = np.linalg.norm(y_b - u_true, axis=0) / np.sqrt(mean_u_true_sq)
exceeded = np.where(E >= e)[0]
n_valid  = exceeded[0] if exceeded.size else n_test
t_valid  = n_valid * Δt

# ------------------------------------------------------------------
# 7. Results
# ------------------------------------------------------------------
print("--------------------------------------------------")
print("LEVEL 1 RESULTS SUMMARY (sine wave):")
print(f"N = {N}, Δt = {Δt}, n_T = {n_T}, n_test = {n_test}")
print(f"max |eig(ρ W)|            : {ρ * np.max(np.abs(np.linalg.eigvals(W))):.4f}")
print(f"Train MSE (one-step)      : {mse_train:.3e}")
print(f"Test MSE (one-step)       : {mse_onestep:.3e}")
print(f"Persistence baseline MSE  : {mse_persist:.3e}")
print(f"Iterative: max E(t)       : {E.max():.3e}   (tolerance e = {e})")
print(f"Iterative: t_valid        : {t_valid:.1f} of {n_test * Δt:.1f} time units")
print("--------------------------------------------------")
