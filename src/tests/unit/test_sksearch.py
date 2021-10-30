"""
Unit tests for sksearch.py

Copyright 2021 Jerrad M. Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

import unittest

import numpy as np

import sksearch


def random_solutions(a, n,
                     rng=np.random.default_rng(),
                     shuffle=True,
                     mutation=2):
    solutions = []
    for _ in range(n):
        if shuffle:
            a0 = rng.choice(a, size=len(a), replace=False)
            if mutation:
                a0 = sksearch._mutate(a0, mutation, rng)

        elif mutation:
            a0 = sksearch._mutate(a, mutation, rng)

        else:
            raise ValueError('shuffle can not be False while mutation is 0')

        solutions.append(a0)

    return solutions


def square_root2(x):
    x = x.flatten()
    return np.abs(x - 1.4142135623730951)


class TestPSO(unittest.TestCase):
    def test_square_root2(self):
        rng = np.random.default_rng(0)
        guess = np.full(1, 100 * rng.random())
        solutions = random_solutions(guess, 100,
                                     rng=rng,
                                     shuffle=False,
                                     mutation=2)

        best, error = sksearch.pso(solutions, square_root2,
                                   vmax=0.5,
                                   max_iter=2000,
                                   max_error=1e-6,
                                   rng=rng,
                                   c1=2,
                                   c2=0.1)

        self.assertAlmostEqual(best[0], 1.4142135623730951, 5)
        self.assertAlmostEqual(square_root2(best), error)

    def test_square_root2_with_multiprocessing(self):
        rng = np.random.default_rng(0)
        guess = np.full(1, 100 * rng.random())
        solutions = random_solutions(guess, 100,
                                     rng=rng,
                                     shuffle=False,
                                     mutation=2)

        best, error = sksearch.pso(solutions, square_root2,
                                   vmax=0.5,
                                   max_iter=3000,
                                   max_error=1e-6,
                                   rng=rng,
                                   c1=2,
                                   c2=0.1,
                                   n_jobs=2)

        self.assertAlmostEqual(best[0], 1.4142135623730951, 5)
        self.assertAlmostEqual(square_root2(best), error)


class TestGA(unittest.TestCase):
    def test_square_root2(self):
        rng = np.random.default_rng(0)
        guess = np.full(1, 100 * rng.random())
        solutions = random_solutions(guess, 100,
                                     rng=rng,
                                     shuffle=False,
                                     mutation=2)

        best, error = sksearch.ga(solutions, square_root2,
                                  max_iter=2000,
                                  eta1=0.01,
                                  max_error=1e-6,
                                  rng=rng)

        self.assertAlmostEqual(best[0], 1.4142135623730951, 5)
        self.assertAlmostEqual(square_root2(best), error)


if __name__ == '__main__':
    unittest.main()
