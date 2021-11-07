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
from functools import lru_cache, singledispatch
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
      eta: An array of shape a.shape specifying the magnitude of the mutations.
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
                      max_error=0,
                      max_iter=1000,
                      early_stopping_rounds=-1,
                      rng=None,
                      verbose=False,
                      p='auto',
                      eta='auto',
                      adaptive_population=False,
                      crossover=uniform_crossover,
                      mutate=default_mutate,
                      selection=fitness_proportional_selection):
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
      early_stopping_rounds: The number of iterations that are allowed to pass
                             without improvement before the function returns.
                             `-1` indicates no early stopping. Default is `-1`.
      p: The first learning rate used by genetic algorithm. Controls the
         frequency of mutations, i.e. the probability that each element in a
         "child" solution will be mutated. May be a float, 'auto', or
         'adaptive'. If set to 'auto', a heuristic is used to set a value for
         `p`. If set to 'adaptive', a heurstic is used to set an initial value
         and shrinking is applied as the number of iterations increase.
      eta: The second learning rate used by genetic algorithm. Controls the
           magnitude of mutations, where higher values of eta correspond to
           higher magnitudes. May be a float, 'auto', or 'adaptive'. If set to
           'auto', a heuristic is used to set a value for `eta`. If set to
           'adaptive', a heurstic is used to set an initial value and shrinking
           is applied as the number of iterations increase.
      adaptive_population: Set to `True` to shrink the population size as the
                           number of iterations increase.
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
    pop_size_min = math.sqrt(pop_size0)

    if eta == 'auto':
        eta_upper = np.std(old_population, axis=0)
        eta_lower = eta_upper * 0.1
        eta0 = np.geomspace(eta_upper, eta_lower, max_iter)[int(max_iter / 2)]

    elif eta == 'adaptive':
        eta0 = np.std(old_population, axis=0)
        eta_min = eta0 * 0.1

    else:
        eta0 = eta

    if p == 'auto':
        p0 = 1 / old_population.shape[1]

    elif p == 'adaptive':
        p0 = 1
        p_min = 1 / old_population.shape[1] / 2

    else:
        p0 = p

    p1 = p0
    eta1 = eta0
    var_error0 = None
    historical_best_solution = None
    historical_min_error = np.inf
    last_improvement = 0
    for iteration in range(max_iter):
        error = loss(old_population)
        iteration_min_error = np.min(error)
        if iteration_min_error < historical_min_error:
            historical_min_error = iteration_min_error
            historical_best_solution = old_population[np.argmin(error)]
            last_improvement = 0

        else:
            last_improvement += 1

        if verbose:
            msg = f'iteration: {iteration} error: {historical_min_error} p: {p1} eta: {eta1} pop_size: {pop_size1}'
            print(msg)

        min_index = np.argmin(error)
        if error[min_index] <= max_error:
            return old_population[min_index], error[min_index]

        if early_stopping_rounds != -1 and last_improvement > early_stopping_rounds:
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
        var_error1 = np.var(error)
        if var_error0 and var_error1 / var_error0 < 0.1 or iteration > math.sqrt(max_iter):
            if p == 'adaptive':
                p1 = _adapt(p0, p_min, iteration, max_iter)

            if eta == 'adaptive':
                eta1 = _adapt(eta0, eta_min, iteration, max_iter)

            if adaptive_population:
                pop_size1 = int(_adapt(pop_size0, pop_size_min, iteration, max_iter))

        elif not var_error0:
            var_error0 = var_error1

    return historical_best_solution, historical_min_error


@singledispatch
def _adapt(x0, x_min, curr_iter, max_iter):
    x1 = _calc_param_space(x0, x_min, max_iter)[curr_iter]
    return x1


@_adapt.register
def _adapt_array(x0: np.ndarray, x_min, curr_iter, max_iter):
    x0_bytes = x0.tobytes()
    x_min_bytes = x_min.tobytes()
    x1 = _calc_param_space_array(x0_bytes, x_min_bytes, x0.dtype, max_iter)[curr_iter]
    return x1


@lru_cache
def _calc_param_space(x0, x_min, max_iter):
    return np.linspace(x0, x_min, max_iter)


@lru_cache
def _calc_param_space_array(x0_bytes, x_min_bytes, dtype, max_iter):
    x0 = np.frombuffer(x0_bytes, dtype=dtype)
    x_min = np.frombuffer(x_min_bytes, dtype=dtype)
    return np.linspace(x0, x_min, max_iter)


ga = genetic_algorithm


def random_restarts(loss, guesses, search_func, *args,
                    rng=None,
                    verbose=False,
                    restarts=2,
                    **kwargs):
    """
    Minimize a loss function using an arbitrary search function with
    random restarts.

    Args:
      loss: The loss function to be minimized. Accepts objects of the
            same type as guesses and returns a 1-D ndarray of error scores,
            where lower scores are better.
      guesses: A 2-D array-like object containing candidate solutions to the
               search problem. Should be compatible with numpy.ndarray.
               This argument may also be a function that accepts an instance
               of numpy.random.Generator and returns guesses.
      search_func: The search function to call.
      *args: Additional positional arguments to `search_func`.
      rng: An instance of numpy.random.Generator. If not given, a new Generator
           will be created.
      verbose: Set to `True` to print the error on each iteration. Default
               is `False`.
      restarts: The number of restarts to perform. Default is `2`.
      **kwargs: Additional keyword arguments to `search_func`.

    Returns:
      A tuple of (best_solution, error).

    """

    if not rng:
        rng = np.random.default_rng()

    best_solution = None
    best_error = np.inf
    for i in range(restarts):
        if verbose:
            print(f'restart: {i}')

        try:
            guesses0 = guesses(rng)

        except TypeError:
            guesses0 = guesses

        solution, error = search_func(loss, guesses0, *args,
                                      rng=rng,
                                      verbose=verbose,
                                      **kwargs)

        if error < best_error:
            best_solution = solution
            best_error = error

    return best_solution, best_error
