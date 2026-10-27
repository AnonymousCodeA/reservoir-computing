# Toy ESN on a sine wave -- notation follows that on the paper by Fleddermann, Herzog & Parlitz,
# Chaos 35, 053147 (2025).
#
#   z_m = [b_in, u_m]                      reservoir input (bias + driving signal)
#   s_m = (1-a) s_{m-1} + a tanh(nu W_in z_m + rho W s_{m-1})        Eq. (1)
#   x_m = [b_out, s_m, u_m]                extended state vector
#   y_m = W_out x_m  (~ u_{m+1})
#   W_out = Y X^T (X X^T + beta I)^-1      Eq. (2)  (Tikhonov regularization)
#   E(t) = ||u(t) - u_true(t)|| / <||u_true||^2>_t^(1/2)             Eq. (3)
#   t_valid = max t such that E(t) < e
import numpy as np

# ---- hyperparameters ----
N     = 100      # node number
eps   = 0.1      # sparseness (fraction of nonzero entries of W)
rho   = 0.95     # spectral radius
nu    = 1.0      # input scaling
alpha = 0.3      # leaking rate
beta  = 1e-6     # regularization constant
dt    = 0.1      # step size Delta t
e     = 0.2      # error tolerance for valid time
b_in, b_out = 1.0, 1.0

T_trans, T_train, T_test = 100, 2000, 500    # steps (transient, training, test)
rng = np.random.default_rng(42)

# ---- data: sine, rescaled to [0, 1]  ----
M = T_trans + T_train + T_test + 1
u = 0.5 * (1 + np.sin(dt * np.arange(M)))    # u_m, shape (M,)
D = 1                                         # dimension of u

# ---- random matrices ----
W_in = rng.uniform(-1, 1, (N, 1 + D))         # couples z_m to the reservoir
W = rng.uniform(-1, 1, (N, N))
W[rng.random((N, N)) > eps] = 0.0             # sparsify
W /= np.max(np.abs(np.linalg.eigvals(W)))     # spectral radius of unity (rho enters in Eq. 1)

def update(s_prev, u_m):
    """Eq. (1): leaky-integrator update."""
    z_m = np.concatenate(([b_in], np.atleast_1d(u_m)))
    return (1 - alpha) * s_prev + alpha * np.tanh(nu * W_in @ z_m + rho * W @ s_prev)

def extended(s_m, u_m):
    return np.concatenate(([b_out], s_m, np.atleast_1d(u_m)))

# ---- transient + training: drive reservoir with true u, record x_m and target u_{m+1} ----
s = rng.random(N)                              # s_0 uniform in [0, 1)^N, as in the paper
X = np.zeros((1 + N + D, T_train))
Y = np.zeros((D, T_train))
for m in range(T_trans + T_train):
    s = update(s, u[m])
    if m >= T_trans:                           # states during transient are not recorded
        k = m - T_trans
        X[:, k] = extended(s, u[m])
        Y[:, k] = u[m + 1]

# ---- Eq. (2): W_out = Y X^T (X X^T + beta I)^-1, solved without explicit inverse ----
A = X @ X.T + beta * np.eye(X.shape[0])
W_out = np.linalg.solve(A, X @ Y.T).T

mse_train = np.mean((W_out @ X - Y) ** 2)

# ---- test (a): one-step prediction, reservoir driven by the true signal ----
m0 = T_trans + T_train                         # index of first test input
s_a = s.copy()
y_a = np.zeros((D, T_test))
y_true = np.zeros((D, T_test))
for k in range(T_test):
    s_a = update(s_a, u[m0 + k])
    y_a[:, k] = W_out @ extended(s_a, u[m0 + k])
    y_true[:, k] = u[m0 + k + 1]
mse_onestep = np.mean((y_a - y_true) ** 2)
mse_persist = np.mean((u[m0:m0 + T_test] - y_true[0]) ** 2)   # baseline y_m = u_m

# ---- test (b): iterative (closed-loop) prediction, y_m fed back as u_{m+1} ----
s_b = s.copy()
y_b = np.zeros((D, T_test))
u_in = u[m0]                                   # only the first input is true data
for k in range(T_test):
    s_b = update(s_b, u_in)
    y_b[:, k] = W_out @ extended(s_b, u_in)
    u_in = y_b[:, k]

# Eq. (3): normalized error and valid time
E = np.linalg.norm(y_b - y_true, axis=0) / np.sqrt(np.mean(np.sum(y_true ** 2, axis=0)))
exceeded = np.where(E >= e)[0]
steps_valid = exceeded[0] if exceeded.size else T_test
t_valid = steps_valid * dt

print("LEVEL 1 RESULTS (paper notation)")
print(f"max |eig(rho W)|             : {rho * np.max(np.abs(np.linalg.eigvals(W))):.4f}")
print(f"train MSE (one-step)         : {mse_train:.3e}")
print(f"test MSE (one-step)          : {mse_onestep:.3e}")
print(f"persistence baseline MSE     : {mse_persist:.3e}")
print(f"iterative: max E(t)          : {E.max():.3e}   (tolerance e = {e})")
print(f"iterative: t_valid           : {t_valid:.1f} of {T_test * dt:.1f} time units")
