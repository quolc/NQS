import numpy as np
import TFI_mcmc, TFI_exact
import sys

def optimization(N, alpha, h, iteration):
    tfi_exact = TFI_exact.TFI_exact(N, alpha, h)
    tfi_mcmc = TFI_mcmc.TFI_MCMC(N, alpha, h, samples=1000)

    # initialize by random
    a = ((np.random.rand(N) - 0.5) * 0.01).astype(np.complex)
    b = ((np.random.rand(N * alpha) - 0.5) * 0.01).astype(np.complex)
    w = ((np.random.rand(N, N * alpha) - 0.5) * 0.01).astype(np.complex)

    # optimization loop
    s_dim = N + N * alpha + N * N * alpha
    mat_delta = np.diag([1] * s_dim) # kronecker delta (identity matrix)

    for p in range(1, iteration+1):
        s, f, eave = tfi_exact.calcsf(a, b, w)
        # s, f = tfi_mcmc.calcsf(a, b, w)
        # _, _, eave = tfi_exact.calcsf(a, b, w) # exact calculation of eave

        # output average energy
        if p % 1 == 0:
            print("{}\t{}".format(p, np.real(eave)))

        # ToDo: currently rank deficience occurs when using explicit regularization
        # lmd = max(100 * (0.9 ** p), 1e-4)
        # s_reg = s + lmd * mat_delta * s
        # sol = np.linalg.solve(s_reg, f) # S^{-1} F

        # Moore-Penrose pseudo-inverse works correctly.
        sol = np.dot(np.linalg.pinv(s), f)

        # ToDo: what is appropriate value for scaling parameter gamma?
        gamma = 0.001
        a -= gamma * sol[0 : N]
        b -= gamma * sol[N : N + N * alpha]
        dw = np.reshape(-gamma * sol[N + N * alpha :], (N, N * alpha))
        # sudden change of w is detected
        if np.max(np.abs(dw)) > 1:
            print(sorted(np.reshape(np.abs(s), (s_dim*s_dim,))))
            sys.exit()
        w += dw


if __name__ == "__main__":
    optimization(N=4, alpha=2, h=0, iteration=2000)
