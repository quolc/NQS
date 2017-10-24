import numpy as np
import TFI_mcmc, TFI_exact

N = 4
alpha = 1
h = 10

seed = 0

exact = TFI_exact.TFI_exact(N, alpha, h)
mcmc = TFI_mcmc.TFI_MCMC(N, alpha, h, samples=1000000, seed=seed)

a = np.array([1 + 1j, 1 + 1j, 1 + 1j, 1 + 1j])
b = np.array([1 + 1j, 1 + 1j, 1 + 1j, 1 + 1j])
w = np.array([[1 + 1j, 1 + 1j, 1 + 1j, 1 + 1j],
              [1 + 1j, 1 + 1j, 1 + 1j, 1 + 1j],
              [1 + 1j, 1 + 1j, 1 + 1j, 1 + 1j],
              [1 + 1j, 1 + 1j, 1 + 1j, 1 + 1j], ]) * 0.01

exact_s, exact_f = exact.calcsf(a, b, w)
mcmc_s, mcmc_f = mcmc.calcsf(a, b, w)

# print(exact_s)
# print(mcmc_s)
# print(abs((mcmc_s-exact_s)/exact_s))
