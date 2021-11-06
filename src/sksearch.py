"""
scikit-search aims to be a production-quality, state-of-the-art library
of heuristics for solving general search and optimization problems.
It implements standard approaches to describing search problems and
solution spaces such that multiple solvers can be combined and applied
to the same problem with a minimum effort on the part of the caller.

Copyright 2021 Jerrad Michael Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

import os
import math
from functools import partial
from multiprocessing import Pool

import numpy as np


def particle_swarm_optimization(loss, guesses,
                                c1=2,
                                c2=2,
                                vmax=2,
                                max_error=0,
                                max_iter=1000,
                                n_jobs=1,
                                rng=None,
                                verbose=False):
    """
    Minimize a loss function using particle swarm optimization.

    Args:
      loss: The loss function to be minimized. Accepts objects of the
            same type as guesses and returns a 1-D ndarray of error scores,
            where lower scores are better.
      guesses: A 2-D array-like object containing candidate solutions to the
               search problem. Should be compatible with numpy.ndarray.
      c1: Personal best learning rate. Should be a positive float. Default is 2.
      c2: Global best learning rate. Should be a positive float. Default is 2.
      vmax: Maximum absolute velocity. Should be a positive float. Default is 2.
      max_error: Maximum error score required for early stopping. Defaults to
                 `0`.
      max_iter: Maximum number of iterations before the function returns.
                Defaults to `1000`.
      rng: An instance of numpy.random.Generator. If not given, a new Generator
           will be created.
      n_jobs: Number of processes to use when computing the loss function
              on each possible function. Set to `1` by default.
      verbose: Set to `True` to print the error on each iteration. Default
               is `False`.

    Returns:
      A tuple of (best_solution, error).

    """

    def pso_(pool):
        nonlocal rng
        nonlocal guesses

        if not rng:
            rng = np.random.default_rng()

        historical_best_solution = None
        historical_min_error = np.inf
        guesses = np.array(guesses)
        pbest = guesses
        pbest_error = np.full(len(guesses), np.inf)
        velocity = np.zeros((len(guesses),) + guesses[0].shape)
        for iteration in range(max_iter):
            gbest = None
            gbest_error = np.inf
            if pool:
                split_guesses = np.split(guesses, len(guesses))
                error = np.array(pool.map(loss, split_guesses)).flatten()

            else:
                error = loss(guesses)

            min_index = np.argmin(error)
            gbest_error = error[min_index]
            gbest = guesses[min_index]
            if gbest_error < historical_min_error:
                historical_best_solution = gbest
                historical_min_error = gbest_error

            if verbose:
                print(f'iteration: {iteration} error: {historical_min_error}')

            if gbest_error <= max_error:
                return gbest, gbest_error

            pbest_filter = error < pbest_error
            pbest_error = np.where(pbest_filter, error, pbest_error)
            pbest_filter = np.resize(pbest_filter, pbest.shape)
            pbest = np.where(pbest_filter, guesses, pbest)
            velocity = (velocity
                        + c1
                        * rng.random(velocity.shape)
                        * (pbest - guesses)
                        + c2
                        * rng.random(velocity.shape)
                        * (gbest - guesses))

            velocity = np.where(np.abs(velocity) > vmax,
                                vmax * np.sign(velocity),
                                velocity)

            guesses = guesses + velocity

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


pso = particle_swarm_optimization


def uniform_crossover(parent_a, parent_b, rng):
    """
    Defines a uniform crossover operator for genetic algorithm.

    In uniform crossover, every element from each parent has an equal
    probability of being chosen, and every element is chosen independently.

    Args:
      parent_a: A 1-D ndarray to breed with parent_b.
      parent_b: A 1-D ndarray to breed with parent_a.
      rng: An instance of `numpy.random.Generator`.

    Returns:
      An ndarray of shape `parent_a.shape` that is the result of breeding
      parent_a with parent_b.

    """

    return np.where(rng.random(len(parent_a)) > 0.5,
                    parent_a,
                    parent_b)


def default_mutate(a, p, eta, rng):
    """
    Defines the default mutation operator for genetic algorithm.

    In this mutation operator, every element of `a` has a probability `p` of
    being randomly mutated with magnitude `eta`.

    Args:
      a: A 1-D ndarray to mutate.
      p: The probability, as a number between 0 and 1, to mutate each element
         of `a`.
      eta: The magnitude of the mutations, as a float > 0.
      rng: An instance of `numpy.random.Generator`.

    Returns:
      A mutated ndarray with shape `a.shape`.

    """

    mutated = a + rng.random(a.shape) * rng.choice([1, -1], a.shape) * eta
    return np.where(rng.random(a.shape) > p, a, mutated)


def fitness_proportional_selection(population, errors, rng):
    """
    Defines a fitness proportional selection operator for genetic algorithm.

    In fitness proportional selection, each row in `population` has a
    probability of being selected that is linearly proportional to its rank in
    the fitness landscape (which is the inverse of the error landscape).

    Args:
      population: A 2-D ndarray of possible solutions to an optimization
                  problem.
      errors: A 1-D nadarray of errors corresponding to each row in
              `population`.

      rng: An instance of `numpy.random.Generator`.

    Yields:
      Rows selected from `population` as 1-D ndarrays.

    """

    population = population[np.argsort(errors)]
    errors = np.sort(errors)
    selection_probability = np.linspace(100, 0,
                                        num=len(population),
                                        endpoint=False)

    scaling_factor = 1 / np.sum(selection_probability)
    selection_probability = selection_probability * scaling_factor
    fp_correction = 1 - np.sum(selection_probability)
    selection_probability[0] += fp_correction
    indices = np.arange(len(population))

    while True:
        parent_a, parent_b = rng.choice(indices,
                                        size=2,
                                        replace=False,
                                        p=selection_probability)

        yield population[parent_a]
        yield population[parent_b]


def genetic_algorithm(loss, guesses,
                      p='auto',
                      eta='auto',
                      adaptive_population_size=False,
                      crossover=uniform_crossover,
                      mutate=default_mutate,
                      selection=fitness_proportional_selection,
                      max_error=0,
                      max_iter=1000,
                      rng=None,
                      verbose=False):
    """
    Minimize a loss function using genetic algorithm.

    Args:
      loss: The loss function to be minimized. Accepts objects of the
            same type as guesses and returns a 1-D ndarray of error scores,
            where lower scores are better.
      guesses: A 2-D array-like object containing candidate solutions to the
               search problem. Should be compatible with numpy.ndarray.
      max_error: Maximum error score required for early stopping. Defaults to
                 `0`.
      max_iter: Maximum number of iterations before the function returns.
                Defaults to `1000`.
      p: The first learning rate used by genetic algorithm. Controls the
         frequency of mutations, i.e. the probability that each element in a
         "child" solution will be mutated. If None (the default), p is
         calculated as `1 / guesses.shape[1]`.
      eta: The second learning rate used by genetic algorithm. Controls the
           magnitude of mutations, where higher values of eta correspond to
           higher magnitudes. Defaults to `2`.
      crossover: A function that defines the genetic crossover operator. Takes
                 three arguments: parent_a and parent_b, both of which are
                 ndarrays of size `guesses.shape[1]`, and `rng`. Returns an
                 ndarray of size `guesses.shape[1]`. Defaults to
                 `uniform_crossover`.
      mutate: A function that defines the genetic mutation operator. Takes
              four arguments: an ndarray of shape `guesses.shape[1]`, `p`,
              `eta`, and `rng`.
      selection: A function that defines the genetic selection operator. Takes
                 three arguments: a 2-D ndarray of guesses, a 1-D ndarray of
                 errors corresponding to `guesses`, and `rng`. Yields individual
                 (1-D) rows from `guesses`. Defaults to
                 `fitness_proportional_selection`.
      rng: An instance of numpy.random.Generator. If not given, a new Generator
           will be created.
      verbose: Set to `True` to print the error on each iteration. Default
               is `False`.

    Returns:
      A tuple of (best_solution, error).

    """

    if not rng:
        rng = np.random.default_rng()

    old_population = np.array(guesses)
    pop_size0 = len(old_population)
    pop_size1 = pop_size0
    if eta in ('auto', 'adaptive'):
        eta0 = np.min(np.std(old_population, axis=0))

    else:
        eta0 = eta

    if p == 'auto':
        p0 = 1 / old_population.shape[1]

    elif p == 'adaptive':
        p0 = 1

    else:
        p0 = p

    p1 = p0
    eta1 = eta0
    _adapt0 = partial(_adapt, _order_of_magnitude(max_iter))

    mean_error0 = None
    var_error0 = None
    historical_best_solution = None
    historical_min_error = np.inf
    for iteration in range(max_iter):
        error = loss(old_population)
        iteration_min_error = np.min(error)
        if iteration_min_error < historical_min_error:
            historical_min_error = iteration_min_error
            historical_best_solution = old_population[np.argmin(error)]

        if verbose:
            msg = f'iteration: {iteration} error: {historical_min_error} p: {p1} eta: {eta1} pop_size: {pop_size1}'
            print(msg)

        min_index = np.argmin(error)
        if error[min_index] <= max_error:
            return old_population[min_index], error[min_index]

        selector = selection(old_population, error, rng)
        new_population = []
        while len(new_population) < pop_size1:
            parent_a = next(selector)
            parent_b = next(selector)
            kid_a = crossover(parent_a, parent_b, rng)
            kid_a = mutate(kid_a, p1, eta1, rng)
            new_population.append(kid_a)

        old_population = np.array(new_population)
        mean_error1 = np.mean(error)
        var_error1 = np.var(error)
        if var_error0 is None:
            var_error0 = var_error1

        if mean_error0 is None and (var_error1 / var_error0 < 0.1 or iteration > math.sqrt(max_iter)):
            mean_error0 = mean_error1

        if mean_error0 and p == 'adaptive':
            p1 = _adapt0(p0, mean_error0, mean_error1, iteration)
            if p1 > p0:
                p1 = p0

        if mean_error0 and eta == 'adaptive':
            eta1 = _adapt0(eta0, mean_error0, mean_error1, iteration)
            if eta1 > eta0:
                eta1 = eta0

        if mean_error0 and adaptive_population_size:
            pop_size1 = int(_adapt0(pop_size0, mean_error0, mean_error1, iteration))
            if pop_size1 > pop_size0:
                pop_size1 = pop_size0

            elif pop_size1 < math.sqrt(pop_size0):
                pop_size1 = int(math.ceil(math.sqrt(pop_size0)))

    return historical_best_solution, historical_min_error


def _adapt(order, x0, orig_error, curr_error, iteration):
    return (curr_error / orig_error + 10 ** order / iteration) / 2 * x0


def _order_of_magnitude(x):
    return math.floor(math.log(x, 10))


ga = genetic_algorithm
