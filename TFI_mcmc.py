import random


class tfi_mcmc:

    def __init__(self, n, alpha, h, seed=None):
        self.n = n
        self.alpha = alpha
        self.h = h
        if seed is not None:
            random.seed(seed)


if __name__ == "__main__":
    pass
