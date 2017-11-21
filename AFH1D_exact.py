import numpy as np
from functools import reduce

class AFH1D_exact:

    def __init__(self, N, alpha, j):
        """
        :param N: number of visible units (i.e., number of spins)
        :param alpha: hidden unit density
        :param j: coupling constant (>0 for anti-ferromagnetism)
        """
        self.N = N
        self.M = alpha * N
        self.j = j

    def calcsf(self, a, b, w):
        """
        calculate (S, F) for a wave function given as (a, b, W) by exact calculation
        :param a: bias for visible units
        :param b: bias for hidden units
        :param w: interaction coefficients
        :return: tuple (S, F)
        """

        state = np.zeros((2**self.N,), dtype=np.complex)

        # explicitly construct Hamiltonian
        # deprecated because of memory cost O(2^self.N * 2^self.N)
        #
        # H = np.zeros((2**self.N, 2**self.N), dtype=np.complex)
        # # J * s_x^i s_x^(i+1)
        # for i in range(0, self.N):
        #     sx = np.array([[0,1], [1,0]])        # sigma x
        #     sy = np.array([[0,-1.0j], [1.0j,0]]) # sigma y
        #     sz = np.array([[1,0], [0,-1]])       # sigma z
        #     se = np.array([[1,0], [0,1]])        # identity
        #     # e * ... * sx * sx * ... * e
        #     x_factors = [se] * self.N
        #     x_factors[i] = sx
        #     x_factors[(i + 1) % self.N] = sx
        #     x_term = reduce(np.kron, x_factors)
        #     # e * ... * sy * sy * ... * e
        #     y_factors = [se] * self.N
        #     y_factors[i] = sy
        #     y_factors[(i + 1) % self.N] = sy
        #     y_term = reduce(np.kron, y_factors)
        #     # e * ... * sz * sz * ... * e
        #     z_factors = [se] * self.N
        #     z_factors[i] = sz
        #     z_factors[(i+1)%self.N] = sz
        #     z_term = reduce(np.kron, z_factors)
        #
        #     H += self.j * x_term + self.j * y_term + self.j * z_term

        # {O_k|psi>} (k=1...w_n)
        Opsi = np.zeros((2**self.N, self.N + self.M + self.N * self.M), dtype=np.complex)
        Hpsi = np.zeros((2**self.N, ), dtype=np.complex)

        # calculate vector representation of the state
        for i_s in range(0, 2**self.N):
            # LSB: i=0, MSB: i=N-1
            s = np.array([((i_s >> i)&1)*2-1 for i in range(self.N)], dtype=np.float32)

            # wave function
            theta = b + np.dot(s, w)
            psi = np.exp(np.dot(a, s)) * np.prod(2 * np.cosh(theta))
            state[i_s] = psi # psi(s) = <s|psi>

            # O|psi>
            O_a = s
            O_b = np.tanh(theta)
            O_w = np.outer(s, np.tanh(theta))
            Opsi[i_s] = np.concatenate((O_a, O_b, np.ndarray.flatten(O_w))) * psi

            # H|psi> = \sum_s H|s> <s|psi>
            for i in range(0, self.N):
                # x_i * x_(i+1)
                Hpsi[i_s ^ ((1 << i) + (1 << (i+1) % self.N))] += self.j * psi
                # y_i * y_(i+1)
                Hpsi[i_s ^ ((1 << i) + (1 << (i+1) % self.N))] -= self.j * s[i] * s[(i+1) % self.N] * psi
                # z_i * z_(i+1)
                Hpsi[i_s] += self.j * s[i] * s[(i+1) % self.N] * psi

        amp = np.dot(np.conj(state), state) # <psi|psi>

        Oave = np.dot(np.conj(state), Opsi) / amp
        OOave = np.dot(np.conj(Opsi).T, Opsi) / amp # <psi|O_{k}†O_{k'}|psi>
        S = OOave - np.outer(np.conj(Oave), Oave) # (S4)

        # Hpsi = np.dot(H, state) # = E_loc|psi>
        Eave = np.dot(np.conj(state), Hpsi) / amp
        EOave = np.dot(Hpsi, np.conj(Opsi)) / amp
        F = EOave - Eave * np.conj(Oave)

        # debug
        # print('wave function (exact):')
        # print(np.real(np.conj(state) * state / amp))
        # print()

        return (S, F, Eave)


if __name__ == "__main__":
    # debug
    afh1d_exact = AFH1D_exact(N=4, alpha=1, j=1)
    a = np.array([1+1j, 1+1j, 1+1j, 1+1j])
    b = np.array([1+1j, 1+1j, 1+1j, 1+1j])
    w = np.array([[1+1j, 1+1j, 1+1j, 1+1j],
                  [1+1j, 1+1j, 1+1j, 1+1j],
                  [1+1j, 1+1j, 1+1j, 1+1j],
                  [1+1j, 1+1j, 1+1j, 1+1j],]) * 0.01
    s, f, _ = afh1d_exact.calcsf(a, b, w)
    print(s)
    print(f)
