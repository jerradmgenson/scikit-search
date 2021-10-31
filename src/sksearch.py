"""
Scikit-Search aims to be a production-quality, state-of-the-art library
of heuristics for solving general search and optimization problems.
It implements standard approaches to describing search problems and
solution spaces such that multiple solvers can be combined and applied
to the same problem with a minimum of programming effort.

Copyright 2021 Jerrad Michael Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

import os
import enum
import random
from multiprocessing import Pool

import numpy as np


class HillClimbingMethods(enum.Enum):
    """
    Possible methods to use for `hill_climbing`.

    Attributes:
      SIMPLE: choose the first successor with a higher score than the
              current state.
      STEEPEST_ASCENT: choose the successor with the highest score.
      STOCHASTIC: randomly choose a successor with a higher score than
                  the current state.

    """

    SIMPLE = enum.auto()
    STEEPEST_ASCENT = enum.auto()
    STOCHASTIC = enum.auto()


def hill_climbing(gen_initial_state,
                  loss,
                  get_successor,
                  max_iterations=1000,
                  method=HillClimbingMethods.STEEPEST_ASCENT,
                  restarts=0,
                  initial_momentum=0,
                  max_momentum=3):
    """
    Find a solution to a search problem using hill climbing.

    Args:
      gen_initial_state: A function that takes no arguments and returns
                         a new search space state.
      loss: A function that accepts search space states and returns a
             score >= 0 and 1 where lower scores are superior to higher
             to scores.
      get_successor: A generator function that accepts a search space
                     state and yields the next state.
      max_iterations: Maximum iterations to perform hill climbing, not
                      counting restarts.
      method: The hill climbing heristic to use. Must be a member of
              `HillClimbingMethods`.
      restarts: The number of times to restart hill climbing from an
                initial state.
      initial_momentum: The amount of momentum to begin with for overcoming
                        local optima. If left to 0, momentum will not be
                        utilized.
      max_momentum: The maximum amount of momentum that can be reached.

    Returns:
      A tuple of (search space state, score) for the best solution that
      is found by the hill climbing algorithm.

    """

    initial_state = gen_initial_state()

    # Global best is the best state found so far, including restarts.
    global_best_state = initial_state
    global_best_score = loss(initial_state)

    # Local best is the best state found so far, not including restarts.
    local_best_state = global_best_state
    local_best_score = global_best_score

    # Restart loop - always enter loop the first time.
    while True:
        momentum = initial_momentum
        while max_iterations != 0:
            best_successor = None
            # Set to -1 to gurantee the first state's score will exceed it.
            best_successor_score = -1
            successors = get_successor(local_best_state)
            if method == HillClimbingMethods.STOCHASTIC:
                successors = list(get_successor(local_best_state))
                random.shuffle(successors)

            for successor in successors:
                successor_score = loss(successor)
                if successor_score > best_successor_score:
                    best_successor = successor
                    best_successor_score = successor_score
                    if method in (HillClimbingMethods.SIMPLE,
                                  HillClimbingMethods.STOCHASTIC):
                        break

            slope = best_successor_score / local_best_score
            if best_successor_score > local_best_score or momentum >= slope:
                momentum = momentum * slope
                if momentum > max_momentum:
                    momentum = max_momentum

                local_best_state = best_successor
                local_best_score = best_successor_score
                max_iterations = max_iterations - 1

            else:
                break

        if local_best_score > global_best_score:
            global_best_state = local_best_state
            global_best_score = local_best_score

        if restarts <= 0:
            # Exit outer (restart) loop.
            break

        restarts = restarts - 1
        local_best_state = gen_initial_state()
        local_best_score = loss(local_best_state)

    return global_best_state, global_best_score


def pso(initial_guesses, loss,
        c1=2,
        c2=2,
        vmax=2,
        max_error=0,
        max_iter=1000,
        rng=None,
        n_jobs=1,
        verbose=False):
    """
    Minimize a loss function using particle swarm optimization.

    Args:
      initial_guesses: A nested array-like object containing candidate
                       solutions to the search problem. Should be
                       compatible with numpy.ndarray.
      loss: The loss function to be minimized. Accepts objects of the
            same type as initial_guesses and returns an ndarray of
            error scores >= 0, where lower scores are better.
      c1: Personal best learning rate. Set to 2 by default.
      c2: Global best learning rate. Set to 2 by default.
      vmax: Maximum absolute velocity. Set to 2 by default.
      max_error: Maximum error score required for early stopping. Set to
                 0 by default.
      max_iter: Maximum number of iterations before the function returns.
                Set to 1000 by default.
      rng: An instance of numpy.random.RandomState. Set to default_rng()
           by default.
      n_jobs: Number of processes to use when computing the loss function
              on each possible function. Set to 1 by default.

    Returns:
      A tuple of (best_solution, error).

    """

    def pso_(pool):
        nonlocal rng
        nonlocal initial_guesses

        if not rng:
            rng = np.random.default_rng()

        historical_best_solution = None
        historical_min_error = np.inf
        initial_guesses = np.array(initial_guesses)
        pbest = initial_guesses
        pbest_error = np.full(len(initial_guesses), np.inf)
        velocity = np.zeros((len(initial_guesses),) + initial_guesses[0].shape)
        for _ in range(max_iter):
            gbest = None
            gbest_error = np.inf
            if pool:
                split_guesses = np.split(initial_guesses, len(initial_guesses))
                error = np.array(pool.map(loss, split_guesses)).flatten()

            else:
                error = loss(initial_guesses)

            min_index = np.argmin(error)
            gbest_error = error[min_index]
            gbest = initial_guesses[min_index]
            if gbest_error < historical_min_error:
                historical_best_solution = gbest
                historical_min_error = gbest_error

            if verbose:
                print(f'Error: {historical_min_error}')

            if gbest_error <= max_error:
                return gbest, gbest_error

            pbest_filter = error < pbest_error
            pbest_error = np.where(pbest_filter, error, pbest_error)
            pbest_filter = np.resize(pbest_filter, pbest.shape)
            pbest = np.where(pbest_filter, initial_guesses, pbest)
            velocity = (velocity
                        + c1
                        * rng.random(velocity.shape)
                        * (pbest - initial_guesses)
                        + c2
                        * rng.random(velocity.shape)
                        * (gbest - initial_guesses))

            velocity = np.where(np.abs(velocity) > vmax,
                                vmax * np.sign(velocity),
                                velocity)

            initial_guesses = initial_guesses + velocity

        return historical_best_solution, historical_min_error

    if n_jobs == 1:
        return pso_(None)

    elif n_jobs == -1:
        with Pool(os.cpu_count()) as pool:
            return pso_(pool)

    elif n_jobs > 1:
        with Pool(n_jobs) as pool:
            return pso_(pool)

    else:
        raise ValueError(f'n_jobs must be an int >= 1 (got {n_jobs})')


def ga(initial_guesses, loss,
       max_error=0,
       max_iter=1000,
       eta1=2,
       eta2=2,
       rng=None,
       verbose=False):
    """
    Minimize a loss function using genetic algorithm.

    Args:
      initial_guesses: A nested array-like object containing candidate
                       solutions to the search problem. Should be
                       compatible with numpy.ndarray.
      loss: The loss function to be minimized. Accepts objects of the
            same type as initial_guesses and returns an ndarray of
            error scores >= 0, where lower scores are better.
      max_error: Maximum error score required for early stopping. Set to
                 0 by default.
      max_iter: Maximum number of iterations before the function returns.
                Set to 1000 by default.
      rng: An instance of numpy.random.RandomState. Set to default_rng()
           by default.

    Returns:
      A tuple of (best_solution, error).

    """

    if not rng:
        rng = np.random.default_rng()

    old_population = np.array(initial_guesses)
    historical_best_solution = None
    historical_min_error = np.inf
    for _ in range(max_iter):
        new_population = []
        error = loss(old_population)
        generation_min_error = np.min(error)
        if generation_min_error < historical_min_error:
            historical_min_error = generation_min_error
            historical_best_solution = old_population[np.argmin(error)]

        if verbose:
            print(f'error: {historical_min_error}')

        old_population = old_population[np.argsort(error)]
        error = np.sort(error)
        if error[0] <= max_error:
            return old_population[0], error[0]

        selection_probability = np.linspace(100, 0,
                                            num=len(old_population),
                                            endpoint=False)

        scaling_factor = 1 / np.sum(selection_probability)
        selection_probability = selection_probability * scaling_factor
        fp_correction = 1 - np.sum(selection_probability)
        selection_probability[0] += fp_correction

        indices = np.arange(len(old_population))
        while len(new_population) < len(old_population):
            parent_a, parent_b = rng.choice(indices,
                                            size=2,
                                            replace=False,
                                            p=selection_probability)

            parent_a = old_population[parent_a]
            parent_b = old_population[parent_b]
            kid_a = np.where(rng.random(len(parent_a)) > 0.5,
                             parent_a,
                             parent_b)

            kid_a = _mutate(kid_a, eta=eta1, rng=rng)
            new_population.append(kid_a)
            if len(new_population) < len(old_population):
                kid_b = np.where(rng.random(len(parent_a)) > 0.5,
                                 parent_a,
                                 parent_b)

            kid_b = _mutate(kid_b, eta=eta2, rng=rng)
            new_population.append(kid_b)

        old_population = np.array(new_population)

    return historical_best_solution, historical_min_error


def _mutate(a, eta=2, rng=np.random.default_rng()):
    mutator = rng.random(a.shape) * rng.choice([1, -1], a.shape) * eta
    return a + mutator
