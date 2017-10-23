import numpy as np
from functools import reduce

class TFI_exact:

    def __init__(self, N, alpha, h):
        """
        :param N: number of visible units (i.e., number of spins)
        :param alpha: hidden unit density
        :param h: magnetic field
        """
        self.N = N
        self.M = alpha * N
        self.h = h

    def calcsf(self, a, b, w):
        """
        calculate (S, F) for a wave function given as (a, b, W)
        :param a: bias for visible units
        :param b: bias for hidden units
        :param w: interaction coefficients, w[i,j] represents interaction btw. v_i and h_j
        :return: tuple (S: covariant matrix, F: force vector)
        """

        # number of all parameters
        state = np.zeros((2**self.N,), dtype=np.complex) # |psi>

        # construct Hamiltonian H
        H = np.zeros((2**self.N, 2**self.N), dtype=np.complex)
        for i in range(0, self.N):
            sx = np.array([[0,1], [1,0]])  # sigma x
            sz = np.array([[1,0], [0,-1]]) # sigma z
            se = np.array([[1,0], [0,1]])  # identity
            # e * ... * sx * ... * e
            x_factors = [se]*self.N
            x_factors[i] = sx
            x_term = reduce(np.kron, x_factors)
            H += (-self.h * x_term)
            # e * ... * sz * sz * ... * e
            z_factors = [se]*self.N
            z_factors[i] = sz
            z_factors[(i+1)%self.N] = sz
            z_term = reduce(np.kron, z_factors)
            H += (-z_term)
        #print(H)

        # {O_k|psi>} (k=1...w_n)
        Opsi = np.zeros((2**self.N, self.N + self.M + self.N * self.M), dtype=np.complex)

        # calculate vector representation of the state
        for i_s in range(0, 2**self.N):
            s = np.array([((i_s >> i)&1)*2-1 for i in reversed(range(self.N))], dtype=np.float32)

            # wave function
            theta = b + np.dot(s, w)
            psi = np.exp(np.dot(a, s)) * np.prod(2 * np.cosh(theta))
            state[i_s] = psi # psi(s) = <s|psi>

            O_a = s
            O_b = np.tanh(theta)
            O_w = np.outer(s, np.tanh(theta))
            Opsi[i_s] = np.concatenate((O_a, O_b, np.ndarray.flatten(O_w))) * psi

        amp = np.dot(np.conj(state), state) # <psi|psi>

        Oave = np.dot(np.conj(state), Opsi) / amp
        OOave = np.dot(np.conj(Opsi).T, Opsi) / amp # <psi|O_{k}†O_{k'}|psi>
        S = OOave - np.outer(np.conj(Oave), Oave) # (S4)

        Hpsi = np.dot(H, state) # = E_loc|psi>
        Eave = np.dot(np.conj(state), Hpsi) / amp
        EOave = np.dot(Hpsi, np.conj(Opsi)) / amp
        F = EOave - Eave * np.conj(Oave)

        return (S, F)


if __name__ == "__main__":
    # debug
    tfi_exact = TFI_exact(N=4, alpha=1, h=1)
    a = np.array([1+1j, 1+1j, 1+1j, 1+1j])
    b = np.array([1+1j, 1+1j, 1+1j, 1+1j])
    w = np.array([[1+1j, 1+1j, 1+1j, 1+1j],
                  [1+1j, 1+1j, 1+1j, 1+1j],
                  [1+1j, 1+1j, 1+1j, 1+1j],
                  [1+1j, 1+1j, 1+1j, 1+1j],])
    s, f = tfi_exact.calcsf(a, b, w)
    print(s)
    print(f)
