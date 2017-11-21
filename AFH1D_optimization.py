import numpy as np
import AFH1D_exact
import sys

def optimization(N, alpha, j, iteration):
    afh1d_exact = AFH1D_exact.AFH1D_exact(N, alpha, j)
#    tfi_mcmc = TFI_mcmc.TFI_MCMC(N, alpha, h, samples=1000)

    # initialize by random
    init_range = 0.1
    a = ((np.random.rand(N) - 0.5) * init_range).astype(np.complex)
    a += ((np.random.rand(N) - 0.5) * init_range) * 1.0j
    b = ((np.random.rand(N * alpha) - 0.5) * init_range).astype(np.complex)
    b += ((np.random.rand(N * alpha) - 0.5) * init_range) * 1.0j
    w = ((np.random.rand(N, N * alpha) - 0.5) * init_range).astype(np.complex)
    w += ((np.random.rand(N, N * alpha) - 0.5) * init_range) * 1.0j

    # log
    prev_a = np.copy(a)
    prev_b = np.copy(b)
    prev_w = np.copy(w)

    # optimization loop
    s_dim = N + N * alpha + N * N * alpha
    mat_delta = np.diag([1] * s_dim) # kronecker delta (identity matrix)

    for p in range(1, iteration+1):
        s, f, eave = afh1d_exact.calcsf(a, b, w)
        # s, f = tfi_mcmc.calcsf(a, b, w)
        # _, _, eave = tfi_exact.calcsf(a, b, w) # exact calculation of eave

        # ToDo: currently rank deficience occurs when using explicit regularization
        # lmd = max(100 * (0.9 ** p), 1e-4)
        # s_reg = s + lmd * mat_delta * s
        # sol = np.linalg.solve(s_reg, f) # S^{-1} F

        # Moore-Penrose pseudo-inverse works correctly.
        sol = np.dot(np.linalg.pinv(s), f)

        # output average energy and parameter change
        if p % 1 == 0 and p > 0:
            a_diff = a - prev_a
            b_diff = b - prev_b
            w_diff = w - prev_w
            prev_a = np.copy(a)
            prev_b = np.copy(b)
            prev_w = np.copy(w)
            print("{}\t{}\t{}\t{}\t{}".format(p, np.real(eave),
                                              np.average(np.abs(a_diff)),
                                              np.average(np.abs(b_diff)),
                                              np.average(np.abs(w_diff))))

        # ToDo: what is appropriate value for scaling parameter gamma?
        gamma = 0.005
        a -= gamma * sol[0 : N]
        b -= gamma * sol[N : N + N * alpha]
        dw = np.reshape(-gamma * sol[N + N * alpha :], (N, N * alpha))
        # sudden change of w is detected
        if np.max(np.abs(dw)) > 100:
            # print(sorted(np.reshape(np.abs(s), (s_dim*s_dim,))))
            sys.exit()
        w += dw
    print(a)
    print(b)
    print(w)


if __name__ == "__main__":
    optimization(N=8, alpha=4, j=1, iteration=2000)
