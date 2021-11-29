"""
A library of tools for writing search functions.

Copyright 2021 Jerrad Michael Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

from itertools import repeat
from functools import wraps
from time import time
from os import cpu_count, name as os_name

import numpy as np
from dask.distributed import Client, Future
from joblib import Memory, hash as joblib_hash


def search_algorithm(search_func):
    """
    A decorator to help with writing search functions. It initializes common
    keyword arguments when the user doesn't provide one (n_jobs, client, rng)
    and provides an event loop that checks for stopping criteria so that
    `search_func` doesn't have to.

    Args:
      search_func: A generator function that may accept arbitrary keyword
                   and positional arguments in addition to the required keyword
                   arguments `client` and `rng`. When it wants to check the
                   stopping conditions, it yields a tuple of:
                   (current_iteration, best_solution, error, msg) where `msg`
                   is an arbitrary string to display to the caller when
                   verbose=True.

    Returns:
      A new function that wraps `search_func` and accepts the additional
      keyword arguments `max_error`, `max_iter`, `max_time`, `n_jobs`,
      and `verbose`.

    """

    @wraps(search_func)
    def new_func(*args,
                 max_error=0,
                 max_iter=1000,
                 max_time=-1,
                 n_jobs=1,
                 client=None,
                 rng=None,
                 verbose=False,
                 **kwargs):
        start_time = time()
        if n_jobs == -1:
            n_jobs = cpu_count()

        if rng is None:
            rng = np.random.default_rng()

        owns_client = False
        try:
            if client is None:
                client = Client(n_workers=n_jobs, set_as_default=False)
                owns_client = True

            for iteration, solution, error, msg in search_func(*args,
                                                               max_iter=max_iter,
                                                               client=client,
                                                               rng=rng,
                                                               **kwargs):
                wall_time = time() - start_time
                if max_time != -1 and wall_time > max_time:
                    return solution, error

                if max_error > error:
                    return solution, error

                if iteration > max_iter:
                    return solution, error

                if verbose:
                    print(f'iteration: {iteration} wall time: {round(wall_time, 2)} error: {round(error, 2)} best solution: {solution} {msg}')

        finally:
            if owns_client:
                client.close()

    return new_func


def infinite_count(start=0):
    """
    Returns an iterator that yields integers from `start` to infinity.

    """

    for i, _ in enumerate(repeat(None)):
        yield i + start


def with_cache(func):
    """
    Add a cache dict to func. On subsequent calls to func, an additional
    keyword argument `cache` will be passed that remains identical
    between function calls.

    """

    cache = dict()

    @wraps(func)
    def new_func(*args, **kwargs):
        return func(*args, cache=cache, **kwargs)

    return new_func


@with_cache
def evaluate_solutions(loss, client, solutions, *solution_groups, cache=None):
    """
    Evaluate solutions using the given loss function.

    Args:
      loss: The loss function to be minimized. Accepts objects of the
            same type as guesses and returns a 1-D ndarray of error scores,
            where lower scores are better.
      client: An instance of `dask.distributed.Client`.
      solutions: A 2-D array-like object containing candidate solutions to the
               search problem. Should be compatible with numpy.ndarray.
      *solution_groups: Additional solutions that may be provided in distinct
                        groups.

    Returns:
      A 1-D array of errors resulting from calling `loss` on `solutions`.
      If `solution_groups` is provided, a list of 1-D error arrays will be
      returned, where the length of the list equals len(solution_groups) + 1
      (where the +1 is for `solutions`).

    """

    solution_groups = (solutions,) + solution_groups
    if len(client.cluster.workers) == 1:
        # If n_jobs == 1, it is more efficient to bypass dask and call
        # the loss function directly.
        errors = loss(np.concatenate(solution_groups))

    else:
        # If n_jobs > 1, use dask to distribute the jobs.
        futures = []
        for solutions in solution_groups:
            for solution in solutions:
                solution_hash = joblib_hash(solution)
                if solution_hash in cache:
                    futures.append(cache[solution_hash])

                else:
                    futures.append(client.submit(loss, np.atleast_2d(solution)))

        errors = []
        for solutions in solution_groups:
            for solution, future in zip(solutions, futures):
                if isinstance(future, Future):
                    result = future.result()
                    cache[joblib_hash(solution)] = result
                    errors.append(result)

                else:
                    errors.append(future)

            futures = futures[len(solutions):]

        errors = np.concatenate(errors)

    if len(solution_groups) == 1:
        return errors

    error_groups = []
    for solutions in solution_groups:
        error_groups.append(errors[:len(solutions)])
        errors = errors[len(solutions):]

    return error_groups


def find_best(solutions_errors, *solution_groups, hbest=None):
    """
    Find the best solution in the given solutions.

    Args:
      solutions_errors: A tuple of (solutions, errors).
      *solution_groups: Additional tuples of (solutions, errors) passed
                        in distinct groups.
      hbest: A tuple of (best_solution, best_error).

    Returns:
      A tuple of (best_solution, best_error) from all the solutions provided.

    """

    if solution_groups:
        solutions, errors = list(zip(*solution_groups))
        solutions += (solutions_errors[0],)
        errors += (solutions_errors[1],)
        solutions = np.concatenate(solutions)
        errors = np.concatenate(errors)

    else:
        solutions, errors = solutions_errors

    if hbest:
        solutions = np.concatenate([[hbest[0]], solutions])
        errors = np.concatenate([[hbest[1]], errors])

    min_index = np.argmin(errors)

    return solutions[min_index], errors[min_index]


def _setup_memory(memory):
    if memory == 'default':
        if os_name == 'posix':
            memory = Memory('/dev/shm/__joblib_cache__', verbose=0)

        else:
            memory = Memory('__joblib_cache__', verbose=0)

    elif not isinstance(memory, Memory):
        memory = Memory(str(memory))

    return memory


def uniform_crossover(parent_a, parent_b, rng):
    """
    Defines a uniform crossover operator.

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
    Defines the default mutation operator.

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
    Defines a fitness proportional selection operator.

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
