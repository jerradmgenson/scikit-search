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
                     eta=2):
    solutions = []
    for _ in range(n):
        a0 = sksearch._mutate(a, eta, rng)
        solutions.append(a0)

    return solutions


def square_root2(x):
    x = x[0]
    return np.abs(x**2 - 2)


def system_of_equations(v):
    x = v[0]
    y = v[1]
    z = v[2]

    s1 = equation1(x, y, z)
    s2 = equation2(x, y, z)
    s3 = equation3(x, y, z)

    error = np.abs(s1 - s2)
    error += np.abs(s1 - s3)
    error += np.abs(s2 - s3)

    return error


def equation1(x, y, z):
    return 3 * x + 2 * y - z - 1


def equation2(x, y, z):
    return 2 * x - 2 * y + 4 * z + 2


def equation3(x, y, z):
    return -x + 0.5 * y - z


class TestPSO(unittest.TestCase):
    def test_square_root2(self):
        rng = np.random.default_rng(0)
        guess = np.full(1, 100 * rng.random())
        solutions = random_solutions(guess, 100,
                                     rng=rng,
                                     eta=2)

        best, error = sksearch.pso(solutions, square_root2,
                                   vmax=0.5,
                                   max_iter=2000,
                                   max_error=1e-3,
                                   rng=rng,
                                   c1=2,
                                   c2=0.1)

        self.assertAlmostEqual(best[0], 1.4142135623730951, 2)
        self.assertAlmostEqual(square_root2(best), error)

    def test_square_root2_with_multiprocessing(self):
        rng = np.random.default_rng(0)
        guess = np.full(1, 100 * rng.random())
        solutions = random_solutions(guess, 100,
                                     rng=rng,
                                     eta=2)

        best, error = sksearch.pso(solutions, square_root2,
                                   vmax=0.5,
                                   max_iter=3000,
                                   max_error=1e-3,
                                   rng=rng,
                                   c1=2,
                                   c2=0.1,
                                   n_jobs=2)

        self.assertAlmostEqual(best[0], 1.4142135623730951, 2)
        self.assertAlmostEqual(square_root2(best), error)

    def test_system_of_equations(self):
        rng = np.random.default_rng(0)
        shape = 100, 3
        guesses = np.full(shape, 100) * rng.random(shape)
        guesses = np.where(rng.random(shape) > 0.5, guesses, -guesses)

        best, error = sksearch.pso(guesses, system_of_equations,
                                   max_iter=10000,
                                   max_error=0.01,
                                   c1=1,
                                   c2=1,
                                   vmax=1,
                                   rng=rng)

        self.assertAlmostEqual(error, system_of_equations(best))
        self.assertLessEqual(error, 0.01)


class TestGA(unittest.TestCase):
    def test_square_root2(self):
        rng = np.random.default_rng(0)
        guess = np.full(1, 100 * rng.random())
        solutions = random_solutions(guess, 100,
                                     rng=rng,
                                     eta=2)

        best, error = sksearch.ga(solutions, square_root2,
                                  max_iter=2000,
                                  eta1=0.01,
                                  max_error=1e-3,
                                  rng=rng)

        self.assertAlmostEqual(best[0], 1.4142135623730951, 2)
        self.assertAlmostEqual(square_root2(best), error)

    def test_system_of_equations(self):
        rng = np.random.default_rng(1)
        shape = 100, 3
        guesses = np.full(shape, 100) * rng.random(shape)
        guesses = np.where(rng.random(shape) > 0.5, guesses, -guesses)

        best, error = sksearch.ga(guesses, system_of_equations,
                                  max_iter=1000,
                                  eta1=1,
                                  eta2=2,
                                  max_error=0.02,
                                  rng=rng)

        self.assertAlmostEqual(error, system_of_equations(best))
        self.assertLessEqual(error, 0.02)


if __name__ == '__main__':
    unittest.main()
