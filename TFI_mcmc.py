import random
import numpy as np

class TFI_MCMC:

    def __init__(self, N, alpha, h, samples=100000, seed=None):
        """
        :param N: number of visible units (i.e., number of spins)
        :param alpha: hidden unit density
        :param h: magnetic field
        """
        self.N = N
        self.M = alpha * N
        self.h = h
        self.samples = samples
        if seed is not None:
            random.seed(seed)

        # random initialization of spins only in the constructor
        self.s = np.array([random.randint(0, 1)*2-1 for _ in range(self.N)], dtype=np.float32)

    def calcsf(self, a, b, w):
        """
        calculate (S, F) for a wave function given as (a, b, W) by MCMC
        :param a: bias for visible units
        :param b: bias for hidden units
        :param w: interaction coefficients, w[i,j] represents interaction btw. v_i and h_j
        :return: tuple (S: covariant matrix, F: force vector)
        """

        # initialize by random configuration
        theta = b + np.dot(self.s, w)
        exp_factor = np.exp(np.dot(a, self.s))
        cos_factor = np.prod(2 * np.cosh(theta))

        n_w = self.N + self.M + self.N * self.M
        Oave = np.zeros((n_w,), dtype=np.complex)
        OOave = np.zeros((n_w,n_w), dtype=np.complex)
        Eave = 0
        EOave = np.zeros((n_w,), dtype=np.complex)

        # debug
        wf = np.zeros((2 ** self.N,))

        # stochastic sampling
        for _ in range(self.samples):
            # sampling physical quantities
            Oa = np.array(self.s)
            Ob = np.tanh(theta)
            Ow = np.outer(self.s, np.tanh(theta))
            O = np.concatenate((Oa, Ob, np.ndarray.flatten(Ow)))

            # calculate local energy <S|H|psi> / psi (H = -h \sum_i sx_i - \sum_ij sz_i sz_j)
            E = 0
            for i in range(self.N):
                new_theta = theta - 2 * w[i] * self.s[i]
                new_exp_factor = exp_factor * np.exp(-2*a[i] * self.s[i])
                new_cos_factor = np.prod(2 * np.cosh(new_theta))
                E += (-self.h * (new_exp_factor * new_cos_factor) / (exp_factor * cos_factor))
                E += (-self.s[i] * self.s[(i + 1) % self.N])

            Oave += O
            OOave += np.outer(np.conj(O), O)
            Eave += E
            EOave += E * np.conj(O)

            # debug : wave function sampling
            # i_s = 0
            # for i in range(self.N):
            #     i_s += 2**i * (self.s[i]+1)/2
            # wf[int(i_s)] += 1.0/self.samples

            # Metropolis-Hastings update
            for _ in range(self.N): # max N flip per a Monte Carlo sweep
                flip_at = random.randint(0, self.N-1)
                new_theta = theta - 2 * w[flip_at] * self.s[flip_at]
                new_exp_factor = exp_factor * np.exp(-2 * a[flip_at] * self.s[flip_at])
                new_cos_factor = np.prod(2 * np.cosh(new_theta)) # O(M)
                prob = min(1, abs((new_exp_factor * new_cos_factor) / (exp_factor * cos_factor)) ** 2)

                if random.random() < prob:
                    self.s[flip_at] *= -1
                    theta = new_theta
                    exp_factor = new_exp_factor
                    cos_factor = new_cos_factor

        Oave /= self.samples
        OOave /= self.samples
        Eave /= self.samples
        EOave /= self.samples

        S = OOave - np.outer(np.conj(Oave), Oave)
        F = EOave - Eave * np.conj(Oave)

        # debug
        # print("wave function (MCMC)")
        # print(wf) # wave function (amplitude)
        # print()

        return (S, F, Eave)


if __name__ == "__main__":
    # debug
    tfi_mcmc = TFI_MCMC(N=4, alpha=1, h=10, seed=0)
    a = np.array([1+1j, 1+1j, 1+1j, 1+1j])
    b = np.array([1+1j, 1+1j, 1+1j, 1+1j])
    # a = np.array([0, 0, 0, 0])
    # b = np.array([0, 0, 0, 0])
    w = np.array([[1+1j, 1+1j, 1+1j, 1+1j],
                  [1+1j, 1+1j, 1+1j, 1+1j],
                  [1+1j, 1+1j, 1+1j, 1+1j],
                  [1+1j, 1+1j, 1+1j, 1+1j],]) * 0.01
    s, f = tfi_mcmc.calcsf(a, b, w)
    print(s)
    print(f)
