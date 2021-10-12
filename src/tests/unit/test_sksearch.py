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


def ordering_cost(a):
    swaps = 0
    a0 = [a[0]]
    unsorted = True
    while unsorted:
        unsorted = False
        for x in a[1:]:
            if a0[-1] > x:
                unsorted = True
                swaps = swaps + 1
                a0.append(a0[-1])
                a0[-2] = x

            else:
                a0.append(x)

        a = a0
        a0 = [a[0]]

    return swaps


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
    sign_penalty = np.where(x >= 0, 1, 2)
    return np.abs(x * x - 2) * sign_penalty


class TestPSO(unittest.TestCase):
    def test_square_root2(self):
        rng = np.random.default_rng(0)
        guess = np.full(1, 100 * rng.random())
        solutions = random_solutions(guess, 100,
                                     rng=rng,
                                     shuffle=False,
                                     mutation=2)

        best, distance = sksearch.pso(solutions, square_root2,
                                      vmax=0.5,
                                      max_iter=1000,
                                      rng=rng,
                                      c1=1,
                                      c2=1)

        self.assertAlmostEqual(best, 1.4124678470364231)
        self.assertAlmostEqual(square_root2(best), distance)

    def test_square_root2_with_multiprocessing(self):
        rng = np.random.default_rng(0)
        guess = np.full(1, 100 * rng.random())
        solutions = random_solutions(guess, 100,
                                     rng=rng,
                                     shuffle=False,
                                     mutation=2)

        best, distance = sksearch.pso(solutions, square_root2,
                                      vmax=0.5,
                                      max_iter=1000,
                                      rng=rng,
                                      c1=1,
                                      c2=1,
                                      n_jobs=2)

        self.assertAlmostEqual(best, 1.4124678470364231)
        self.assertAlmostEqual(square_root2(best), distance)


if __name__ == '__main__':
    unittest.main()
