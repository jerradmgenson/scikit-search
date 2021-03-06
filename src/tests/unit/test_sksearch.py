"""
Unit tests for sksearch.py

Copyright 2021 Jerrad M. Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

import os
import sys
import time
import unittest
from pathlib import Path
from functools import lru_cache

import numpy as np
from joblib import Parallel
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

import sksearch
import searchlib as sl

TEST_DATA = Path(__file__).parent.parent / 'data'


def random_solutions(a, n,
                     rng=np.random.default_rng(),
                     eta=2):
    solutions = []
    for _ in range(n):
        a0 = sl.default_mutate(a, 1, eta, rng)
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

        best, error = sksearch.pso(square_root2, solutions,
                                   vmax=0.5,
                                   max_iter=2000,
                                   max_error=1e-3,
                                   rng=rng,
                                   verbose=True,
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

        best, error = sksearch.pso(square_root2, solutions,
                                   vmax=0.5,
                                   max_iter=3000,
                                   max_error=1e-3,
                                   rng=rng,
                                   c1=2,
                                   c2=0.1,
                                   verbose=True,
                                   n_jobs=2)

        self.assertAlmostEqual(best[0], 1.4142135623730951, 2)
        self.assertAlmostEqual(square_root2(best), error)

    def test_system_of_equations(self):
        rng = np.random.default_rng(0)
        shape = 100, 3
        guesses = np.full(shape, 100) * rng.random(shape)
        guesses = np.where(rng.random(shape) > 0.5, guesses, -guesses)

        best, error = sksearch.pso(system_of_equations, guesses,
                                   max_iter=10000,
                                   max_error=0.2,
                                   c1=2,
                                   c2=1,
                                   vmax=12,
                                   verbose=True,
                                   rng=rng)

        self.assertAlmostEqual(error, system_of_equations(np.array([best])))
        self.assertLessEqual(error, 0.2)

    def test_decision_tree(self):
        rng = np.random.default_rng(0)
        shape = 100, 5
        guesses = np.full(shape, 4) * rng.random(shape)
        guesses = np.where(rng.random(shape) > 0.5, guesses, -guesses)

        best, error = sksearch.pso(heart_disease_classifier, guesses,
                                   c1=2,
                                   c2=1,
                                   vmax=12,
                                   rng=rng,
                                   verbose=True,
                                   max_error=0.56)

        self.assertLessEqual(error, 0.56)

    def test_early_stopping(self):
        rng = np.random.default_rng(0)
        shape = 100, 5
        guesses = np.full(shape, 4) * rng.random(shape)
        guesses = np.where(rng.random(shape) > 0.5, guesses, -guesses)

        best, error = sksearch.pso(heart_disease_classifier, guesses,
                                   c1=2,
                                   c2=1,
                                   vmax=12,
                                   rng=rng,
                                   verbose=True,
                                   early_stopping_rounds=2,
                                   max_error=0.56)

        self.assertGreater(error, 0.56)

    def test_max_time(self):
        guesses = np.zeros((100, 10))
        tick = time.time()
        sksearch.pso(lambda x: np.ones(len(x)),
                     guesses,
                     max_iter=100000000,
                     max_error=0,
                     verbose=True,
                     max_time=5)

        tock = time.time()
        self.assertEqual(round(tock - tick), 5.0)

    def test_verbose(self):
        guesses = np.zeros((100, 10))
        try:
            stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            sksearch.pso(lambda x: np.ones(len(x)),
                         guesses,
                         max_iter=2,
                         max_error=0,
                         verbose=True)

        finally:
            sys.stdout.close()
            sys.stdout = stdout

    def test_n_jobs_all_cores(self):
        """
        Test pso `n_jobs` with all cpu cores`

        """

        rng = np.random.default_rng(0)
        shape = 100, 5
        guesses = np.full(shape, 16) * rng.random(shape)
        best, error = sksearch.pso(heart_disease_classifier, guesses,
                                   c1=2,
                                   c2=1,
                                   vmax=12,
                                   n_jobs=-1,
                                   rng=rng,
                                   verbose=True,
                                   max_error=0.56)

        self.assertLessEqual(error, 0.56)

    def test_n_jobs_neg_2_raises_value_error(self):
        """
        Test that calling pso with `n_jobs=-2` raises a ValueError

        """

        rng = np.random.default_rng(0)
        shape = 100, 5
        guesses = np.full(shape, 16) * rng.random(shape)
        with self.assertRaises(ValueError):
            sksearch.pso(heart_disease_classifier, guesses,
                         n_jobs=-2)


class TestGA(unittest.TestCase):
    def test_square_root2(self):
        rng = np.random.default_rng(1)
        guess = np.full(1, 100 * rng.random())
        solutions = random_solutions(guess, 100,
                                     rng=rng,
                                     eta=2)

        best, error = sksearch.ga(square_root2, solutions,
                                  max_iter=2000,
                                  eta=0.5,
                                  max_error=1e-3,
                                  rng=rng)

        self.assertAlmostEqual(best[0], 1.4142135623730951, 2)
        self.assertAlmostEqual(square_root2(best), error)

    def test_system_of_equations(self):
        rng = np.random.default_rng(1)
        shape = 100, 3
        guesses = np.full(shape, 100) * rng.random(shape)
        guesses = np.where(rng.random(shape) > 0.5, guesses, -guesses)

        best, error = sksearch.ga(system_of_equations, guesses,
                                  max_iter=8000,
                                  p=1,
                                  eta=0.18,
                                  max_error=0.009,
                                  rng=rng)

        self.assertAlmostEqual(error, system_of_equations(np.array([best])))
        self.assertLessEqual(error, 0.009)

    def test_decision_tree(self):
        rng = np.random.default_rng(1)
        shape = 100, 5
        guesses = np.full(shape, 4) * rng.random(shape)
        guesses = np.where(rng.random(shape) > 0.5, guesses, -guesses)

        best, error = sksearch.ga(heart_disease_classifier, guesses,
                                  eta=4,
                                  rng=rng,
                                  max_error=0.56)

        self.assertLessEqual(error, 0.56)

    def test_square_root2_adaptive(self):
        rng = np.random.default_rng(1)
        guess = np.full(1, 100 * rng.random())
        solutions = random_solutions(guess, 100,
                                     rng=rng,
                                     eta=2)

        best, error = sksearch.ga(square_root2, solutions,
                                  max_iter=1000,
                                  p='adaptive',
                                  eta='adaptive',
                                  adaptive_population=True,
                                  max_error=1e-3,
                                  rng=rng)

        self.assertAlmostEqual(abs(best[0]), 1.4142135623730951, 2)
        self.assertAlmostEqual(square_root2(best), error)

    def test_system_of_equations_adaptive(self):
        rng = np.random.default_rng(1)
        shape = 100, 3
        guesses = np.full(shape, 5) * rng.random(shape)
        guesses = np.where(rng.random(shape) > 0.5, guesses, -guesses)

        best, error = sksearch.ga(system_of_equations, guesses,
                                  max_iter=2000,
                                  p='adaptive',
                                  eta='adaptive',
                                  adaptive_population=True,
                                  max_error=0.04,
                                  rng=rng)

        self.assertAlmostEqual(error, system_of_equations(np.array([best])))
        self.assertLessEqual(error, 0.04)

    def test_decision_tree_adaptive(self):
        rng = np.random.default_rng(1)
        shape = 100, 5
        guesses = np.full(shape, 8) * rng.random(shape)

        best, error = sksearch.ga(heart_disease_classifier, guesses,
                                  max_iter=1000,
                                  p='adaptive',
                                  eta='adaptive',
                                  adaptive_population=True,
                                  rng=rng,
                                  max_error=0.56)

        self.assertLessEqual(error, 0.56)

    def test_square_root2_auto(self):
        rng = np.random.default_rng(1)
        guess = np.full(1, 100 * rng.random())
        solutions = random_solutions(guess, 100,
                                     rng=rng,
                                     eta=2)

        best, error = sksearch.ga(square_root2, solutions,
                                  max_iter=1000,
                                  p='auto',
                                  eta='auto',
                                  max_error=1e-3,
                                  rng=rng)

        self.assertAlmostEqual(abs(best[0]), 1.4142135623730951, 2)
        self.assertAlmostEqual(square_root2(best), error)

    def test_system_of_equations_auto(self):
        rng = np.random.default_rng(1)
        shape = 100, 3
        guesses = np.full(shape, 10) * rng.random(shape)
        guesses = np.where(rng.random(shape) > 0.5, guesses, -guesses)

        best, error = sksearch.ga(system_of_equations, guesses,
                                  max_iter=3000,
                                  p='auto',
                                  eta='auto',
                                  max_error=0.04,
                                  rng=rng)

        self.assertAlmostEqual(error, system_of_equations(np.array([best])))
        self.assertLessEqual(error, 0.04)

    def test_decision_tree_auto(self):
        rng = np.random.default_rng(0)
        shape = 100, 5
        guesses = np.full(shape, 16) * rng.random(shape)

        best, error = sksearch.ga(heart_disease_classifier, guesses,
                                  max_iter=200,
                                  p='auto',
                                  eta='auto',
                                  rng=rng,
                                  max_error=0.63)

        self.assertLessEqual(error, 0.63)

    def test_square_root2_elitism(self):
        rng = np.random.default_rng(1)
        guess = np.full(1, 100 * rng.random())
        solutions = random_solutions(guess, 100,
                                     rng=rng,
                                     eta=2)

        best, error = sksearch.ga(square_root2, solutions,
                                  max_iter=1000,
                                  p='auto',
                                  eta='auto',
                                  max_error=1e-3,
                                  elitism=True,
                                  rng=rng)

        self.assertAlmostEqual(abs(best[0]), 1.4142135623730951, 2)
        self.assertAlmostEqual(square_root2(best), error)

    def test_time_limit(self):
        guesses = np.zeros((100, 10))
        tick = time.time()
        sksearch.ga(lambda x: np.ones(len(x)),
                    guesses,
                    eta=1,
                    max_iter=100000000,
                    max_error=0,
                    max_time=5)

        tock = time.time()
        self.assertEqual(round(tock - tick), 5.0)

    def test_n_jobs_2(self):
        """
        Test genetic_algorithm with `n_jobs=2`

        """

        rng = np.random.default_rng(0)
        shape = 100, 5
        guesses = np.full(shape, 16) * rng.random(shape)
        best, error = sksearch.ga(heart_disease_classifier, guesses,
                                  max_iter=200,
                                  p='auto',
                                  eta='auto',
                                  n_jobs=2,
                                  rng=rng,
                                  max_error=0.63)

        self.assertLessEqual(error, 0.63)

    def test_n_jobs_all_cores(self):
        """
        Test genetic_algorithm with `n_jobs=-1`

        """

        rng = np.random.default_rng(0)
        shape = 100, 5
        guesses = np.full(shape, 16) * rng.random(shape)
        best, error = sksearch.ga(heart_disease_classifier, guesses,
                                  max_iter=200,
                                  p='auto',
                                  eta='auto',
                                  n_jobs=-1,
                                  rng=rng,
                                  max_error=0.63)

        self.assertLessEqual(error, 0.63)

    def test_early_stopping(self):
        rng = np.random.default_rng(1)
        shape = 100, 3
        guesses = np.full(shape, 10) * rng.random(shape)
        guesses = np.where(rng.random(shape) > 0.5, guesses, -guesses)
        _, error = sksearch.ga(system_of_equations, guesses,
                               max_iter=3000,
                               p='auto',
                               eta='auto',
                               max_error=0.04,
                               early_stopping_rounds=10,
                               rng=rng)

        self.assertGreater(error, 0.04)

    def test_verbose(self):
        guesses = np.zeros((100, 10))
        try:
            stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            sksearch.ga(lambda x: np.ones(len(x)),
                        guesses,
                        eta=1,
                        max_iter=5,
                        max_error=0)

        finally:
            sys.stdout.close()
            sys.stdout = stdout

    def test_n_jobs_neg_2_raises_value_error(self):
        """
        Test that calling ga with `n_jobs=-2` raises a ValueError

        """

        rng = np.random.default_rng(0)
        shape = 100, 5
        guesses = np.full(shape, 16) * rng.random(shape)
        with self.assertRaises(ValueError):
            sksearch.ga(lambda x: np.ones(len(x)),
                        guesses,
                        n_jobs=-2)


class TestRandomRestarts(unittest.TestCase):
    def test_static_guesses(self):
        rng = np.random.default_rng(1)
        guess = np.full(1, 100 * rng.random())
        guesses = random_solutions(guess, 100,
                                   rng=rng,
                                   eta=2)

        best, error = sksearch.random_restarts(square_root2,
                                               guesses,
                                               sksearch.ga,
                                               max_iter=1000,
                                               p='auto',
                                               eta='auto',
                                               max_error=1e-3,
                                               rng=rng)

        self.assertAlmostEqual(abs(best[0]), 1.4142135623730951, 2)
        self.assertAlmostEqual(square_root2(best), error)

    def test_dynamic_guesses(self):
        def gen_guesses(rng):
            guess = np.full(1, 100 * rng.random())
            guesses = random_solutions(guess, 100,
                                       rng=rng,
                                       eta=2)

            return guesses

        rng = np.random.default_rng(1)
        best, error = sksearch.random_restarts(square_root2,
                                               gen_guesses,
                                               sksearch.ga,
                                               max_iter=1000,
                                               p='auto',
                                               eta='auto',
                                               max_error=1e-3,
                                               rng=rng)

        self.assertAlmostEqual(abs(best[0]), 1.4142135623730951, 2)
        self.assertAlmostEqual(square_root2(best), error)


class TestMA(unittest.TestCase):
    def test_square_root2(self):
        rng = np.random.default_rng(1)
        guess = np.full(1, 100 * rng.random())
        solutions = random_solutions(guess, 100,
                                     rng=rng,
                                     eta=2)

        best, error = sksearch.ma(square_root2, solutions,
                                  max_iter=2000,
                                  max_error=1e-3,
                                  rng=rng)

        self.assertAlmostEqual(abs(best[0]), 1.4142135623730951, 2)
        self.assertAlmostEqual(square_root2(best), error)

    def test_system_of_equations(self):
        rng = np.random.default_rng(1)
        shape = 100, 3
        guesses = np.full(shape, 100) * rng.random(shape)
        guesses = np.where(rng.random(shape) > 0.5, guesses, -guesses)
        best, error = sksearch.ma(system_of_equations, guesses,
                                  max_iter=8000,
                                  max_error=0.009,
                                  vmax='auto',
                                  rng=rng)

        self.assertAlmostEqual(error, system_of_equations(np.array([best])))
        self.assertLessEqual(error, 0.009)

    def test_decision_tree(self):
        rng = np.random.default_rng(1)
        shape = 100, 5
        guesses = np.full(shape, 4) * rng.random(shape)
        guesses = np.where(rng.random(shape) > 0.5, guesses, -guesses)

        best, error = sksearch.ma(heart_disease_classifier, guesses,
                                  d=1,
                                  fl=1,
                                  sigma=1,
                                  max_iter=1000,
                                  p=0,
                                  eta=0,
                                  rng=rng,
                                  max_error=0.56)

        self.assertLessEqual(error, 0.56)

    def test_decision_tree_distributed(self):
        rng = np.random.default_rng(1)
        shape = 100, 5
        guesses = np.full(shape, 4) * rng.random(shape)
        guesses = np.where(rng.random(shape) > 0.5, guesses, -guesses)

        best, error = sksearch.ma(heart_disease_classifier, guesses,
                                  d=1,
                                  fl=1,
                                  sigma=1,
                                  max_iter=1000,
                                  p=0,
                                  eta=0,
                                  n_jobs=-1,
                                  rng=rng,
                                  max_error=0.56)

        self.assertLessEqual(error, 0.56)

    def test_scalar_vmax(self):
        rng = np.random.default_rng(1)
        shape = 100, 3
        guesses = np.full(shape, 100) * rng.random(shape)
        guesses = np.where(rng.random(shape) > 0.5, guesses, -guesses)
        best, error = sksearch.ma(system_of_equations, guesses,
                                  max_iter=8000,
                                  max_error=0.009,
                                  vmax=148,
                                  rng=rng)

        self.assertAlmostEqual(error, system_of_equations(np.array([best])))
        self.assertLessEqual(error, 0.009)

    def test_vector_vmax(self):
        rng = np.random.default_rng(1)
        shape = 100, 3
        guesses = np.full(shape, 100) * rng.random(shape)
        guesses = np.where(rng.random(shape) > 0.5, guesses, -guesses)
        best, error = sksearch.ma(system_of_equations, guesses,
                                  max_iter=8000,
                                  max_error=0.009,
                                  vmax=[46.44211451, 6.87179409, 148.86339668],
                                  rng=rng)

        self.assertAlmostEqual(error, system_of_equations(np.array([best])))
        self.assertLessEqual(error, 0.009)

    def test_gmax_is_none(self):
        rng = np.random.default_rng(1)
        guess = np.full(1, 100 * rng.random())
        solutions = random_solutions(guess, 100,
                                     rng=rng,
                                     eta=2)

        best, error = sksearch.ma(square_root2, solutions,
                                  max_iter=2000,
                                  max_error=1e-3,
                                  gmax=None,
                                  rng=rng)

        self.assertAlmostEqual(abs(best[0]), 1.4142135623730951, 2)
        self.assertAlmostEqual(square_root2(best), error)

    def test_auto_eta_and_p(self):
        rng = np.random.default_rng(1)
        shape = 100, 3
        guesses = np.full(shape, 100) * rng.random(shape)
        guesses = np.where(rng.random(shape) > 0.5, guesses, -guesses)
        best, error = sksearch.ma(system_of_equations, guesses,
                                  max_iter=8000,
                                  max_error=0.009,
                                  vmax='auto',
                                  p='auto',
                                  eta='auto',
                                  rng=rng)

        self.assertAlmostEqual(error, system_of_equations(np.array([best])))
        self.assertLessEqual(error, 0.009)


if __name__ == '__main__':
    unittest.main()
