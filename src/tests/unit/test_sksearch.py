"""
Unit tests for sksearch.py

Copyright 2021 Jerrad M. Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

import unittest
from pathlib import Path
from functools import lru_cache

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

import sksearch

TEST_DATA = Path(__file__).parent.parent / 'data'


def random_solutions(a, n,
                     rng=np.random.default_rng(),
                     eta=2):
    solutions = []
    for _ in range(n):
        a0 = sksearch._mutate(a, eta, rng)
        solutions.append(a0)

    return solutions


def square_root2(x):
    return np.abs(x**2 - 2).flatten()


def system_of_equations(m):
    x = m[:, 0]
    y = m[:, 1]
    z = m[:, 2]

    error = np.abs(3 * x + 2 * y - z - 1)
    error += np.abs(2 * x - 2 * y + 4 * z + 2)
    error += np.abs(-x + 0.5 * y - z)

    return error


@lru_cache
def load_heart_disease():
    heart_disease = np.genfromtxt(TEST_DATA / 'heart_disease.csv',
                                  delimiter=',',
                                  missing_values='?',
                                  filling_values='-1')

    X = heart_disease[:, :-1]
    y = heart_disease[:, -1]

    return X, y


def heart_disease_classifier(m, random_state=0):
    X, y = load_heart_disease()
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=random_state)

    errors = []
    for v in m:
        splitter = 'best' if v[0] > 0 else 'random'
        max_depth = int(np.round(np.abs(v[1])))
        if max_depth < 1:
            max_depth = 1

        max_features = int(np.round(np.abs(v[2])))
        if max_features < 1:
            max_features = 1

        elif max_features > X.shape[1]:
            max_features = X.shape[1]

        min_samples_split = int(np.round(np.abs(v[3])))
        if min_samples_split < 2:
            min_samples_split = 2

        elif min_samples_split > len(X_train):
            min_samples_split = len(X_train)

        min_samples_leaf = int(np.round(np.abs(v[4])))
        if min_samples_leaf < 1:
            min_samples_leaf = 1

        elif min_samples_leaf > len(X_train):
            min_samples_leaf = len(X_train)

        dtc = DecisionTreeClassifier(splitter=splitter,
                                     max_depth=max_depth,
                                     max_features=max_features,
                                     min_samples_split=min_samples_split,
                                     min_samples_leaf=min_samples_leaf,
                                     random_state=random_state)

        dtc.fit(X_train, y_train)
        y_pred = dtc.predict(X_test)

        informedness = balanced_accuracy_score(y_test, y_pred, adjusted=True)
        error = np.abs(1 - informedness)
        errors.append(error)

    return np.array(errors)


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
                                   max_error=0.2,
                                   c1=2,
                                   c2=1,
                                   vmax=12,
                                   rng=rng)

        self.assertAlmostEqual(error, system_of_equations(np.array([best])))
        self.assertLessEqual(error, 0.2)

    def test_heart_disease_classifier(self):
        rng = np.random.default_rng(0)
        shape = 100, 5
        guesses = np.full(shape, 4) * rng.random(shape)
        guesses = np.where(rng.random(shape) > 0.5, guesses, -guesses)

        best, error = sksearch.pso(guesses, heart_disease_classifier,
                                   c1=2,
                                   c2=1,
                                   vmax=12,
                                   rng=rng,
                                   max_error=0.56)

        self.assertLessEqual(error, 0.56)


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
                                  max_iter=8000,
                                  eta1=0.18,
                                  eta2=0.25,
                                  max_error=0.006,
                                  rng=rng)

        self.assertAlmostEqual(error, system_of_equations(np.array([best])))
        self.assertLessEqual(error, 0.006)

    def test_heart_disease_classifier(self):
        rng = np.random.default_rng(0)
        shape = 100, 5
        guesses = np.full(shape, 4) * rng.random(shape)
        guesses = np.where(rng.random(shape) > 0.5, guesses, -guesses)

        best, error = sksearch.ga(guesses, heart_disease_classifier,
                                  eta1=4,
                                  eta2=2,
                                  rng=rng,
                                  max_error=0.56)

        self.assertLessEqual(error, 0.56)


if __name__ == '__main__':
    unittest.main()
