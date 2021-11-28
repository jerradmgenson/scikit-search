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
from time import time
from itertools import repeat
from functools import wraps

import numpy as np
from joblib import hash as joblib_hash
from joblib import Memory, Parallel, delayed
from dask.distributed import Future, Client


def search_algorithm(search_func):
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
            n_jobs = os.cpu_count()

        if client is None:
            client = Client(n_workers=n_jobs)

        if rng is None:
            rng = np.random.default_rng()

        for iteration, solution, error, msg in search_func(*args, client=client, rng=rng, **kwargs):
            wall_time = time() - start_time
            if max_time != -1 and wall_time > max_time:
                return solution, error

            if max_error > error:
                return solution, error

            if iteration > max_iter:
                return solution, error

            if verbose:
                print(f'iteration: {iteration} wall time: {round(wall_time, 2)} error: {round(error, 2)} best solution: {solution} {msg}')

    return new_func


def particle_swarm_optimization(loss, guesses,
                                c1=2,
                                c2=2,
                                vmax=2,
                                max_error=0,
                                max_iter=1000,
                                early_stopping_rounds=-1,
                                time_limit=-1,
                                n_jobs=1,
                                memory='default',
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
      early_stopping_rounds: The number of iterations that are allowed to pass
                             without improvement before the function returns.
                             `-1` indicates no early stopping. Default is `-1`.
      time_limit: Amount of time that genetic algorithm is allowed to
                  run in seconds. `-1` means no time limit. Default is `-1`.
      n_jobs: Number of processes to use when evaluating the loss function `-1`
              creates a process for each available CPU. May also be an instance
              of `joblib.Parallel`. Default is `1`.
      memory: Location of joblib cache on the filesystem. May a string, a
              `Path` object, an instance of `joblib.Memory`, None, or
              'default', which uses a (generally good) location specific to
              your operating system.
      rng: An instance of numpy.random.Generator. If not given, a new Generator
           will be created.
      verbose: Set to `True` to print the error on each iteration. Default
               is `False`.

    Returns:
      A tuple of (best_solution, error).

    """

    if time_limit != -1:
        start_time = time.time()

    memory = _setup_memory(memory)
    if memory:
        loss = memory.cache(loss)

    def pso_(parallel):
        nonlocal rng
        nonlocal guesses

        if not rng:
            rng = np.random.default_rng()

        historical_best_solution = None
        historical_min_error = np.inf
        last_improvement = 0
        guesses = np.array(guesses)
        pbest = guesses
        pbest_error = np.full(len(guesses), np.inf)
        velocity = np.zeros((len(guesses),) + guesses[0].shape)
        for iteration in range(max_iter):
            gbest = None
            gbest_error = np.inf
            if parallel:
                splits = np.split(guesses, len(guesses))
                error = np.array(parallel(delayed(loss)(split) for split in splits)).flatten()

            else:
                error = loss(guesses)

            min_index = np.argmin(error)
            gbest_error = error[min_index]
            gbest = guesses[min_index]
            if gbest_error < historical_min_error:
                historical_best_solution = gbest
                historical_min_error = gbest_error
                last_improvement = 0

            else:
                last_improvement += 1

            if verbose:
                print(f'iteration: {iteration} error: {historical_min_error}')

            if gbest_error <= max_error:
                return gbest, gbest_error

            if early_stopping_rounds != -1 and last_improvement > early_stopping_rounds:
                break

            if time_limit != -1 and time.time() - start_time > time_limit:
                break

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

    if isinstance(n_jobs, Parallel):
        return pso_(n_jobs)

    elif n_jobs == 1:
        return pso_(None)

    elif n_jobs == -1:
        with Parallel(n_jobs=os.cpu_count()) as parallel:
            return pso_(parallel)

    elif n_jobs > 1:
        with Parallel(n_jobs=n_jobs) as parallel:
            return pso_(parallel)

    else:
        raise ValueError(f'n_jobs must be an int >= 1 (got {n_jobs})')


pso = particle_swarm_optimization


def _setup_memory(memory):
    if memory == 'default':
        if os.name == 'posix':
            memory = Memory('/dev/shm/__joblib_cache__', verbose=0)

        else:
            memory = Memory('__joblib_cache__', verbose=0)

    elif not isinstance(memory, Memory):
        memory = Memory(str(memory))

    return memory


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
                      time_limit=-1,
                      n_jobs=1,
                      memory='default',
                      rng=None,
                      verbose=False,
                      p='auto',
                      eta='auto',
                      adaptive_population=False,
                      elitism=False,
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
      time_limit: Amount of time that genetic algorithm is allowed to
                  run in seconds. `-1` means no time limit. Default is `-1`.
      n_jobs: Number of processes to use when evaluating the loss function `-1`
              creates a process for each available CPU. May also be an instance
              of `joblib.Parallel`. Default is `1`.
      memory: Location of joblib cache on the filesystem. May a string, a
              `Path` object, an instance of `joblib.Memory`, None, or
              'default', which uses a (generally good) location specific to
              your operating system.
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
                           number of iterations increase. Default is `False`.
      elitism: Set to `True` to allow the best solution from the current
              generation to carry over to the next, unaltered. Default is
              `False`.
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

    if time_limit != -1:
        start_time = time.time()

    memory = _setup_memory(memory)
    if memory:
        loss = memory.cache(loss)

    if not rng:
        rng = np.random.default_rng()

    def ga_(parallel):
        old_population = np.array(guesses)
        pop_size0 = len(old_population)
        pop_size1 = pop_size0
        pop_size_min = math.sqrt(pop_size0)
        if adaptive_population:
            pop_size_space = [int(s) for s in np.linspace(pop_size0, pop_size_min, max_iter)]

        if eta == 'auto':
            eta_upper = np.std(old_population, axis=0)
            eta_lower = eta_upper * 0.1
            eta0 = np.geomspace(eta_upper, eta_lower, max_iter)[int(max_iter / 2)]

        elif eta == 'adaptive':
            eta0 = np.std(old_population, axis=0)
            eta_min = eta0 * 0.1
            eta_space = np.linspace(eta0, eta_min, max_iter)

        else:
            eta0 = eta

        if p == 'auto':
            p0 = 1 / old_population.shape[1]

        elif p == 'adaptive':
            p0 = 1
            p_min = 1 / old_population.shape[1] / 2
            p_space = np.linspace(p0, p_min, max_iter)

        else:
            p0 = p

        p1 = p0
        eta1 = eta0
        var_error0 = None
        historical_best_solution = None
        historical_min_error = np.inf
        last_improvement = 0
        for iteration in range(max_iter):
            if parallel:
                splits = np.split(old_population, len(old_population))
                error = np.array(parallel(delayed(loss)(split) for split in splits)).flatten()

            else:
                error = loss(old_population)

            iteration_min_error = np.min(error)
            if iteration_min_error < historical_min_error:
                historical_min_error = iteration_min_error
                historical_best_solution = old_population[np.argmin(error)]
                last_improvement = 0

            else:
                last_improvement += 1

            if verbose:
                msg = f'iteration: {iteration}/{max_iter} error: {historical_min_error} best solution: {historical_best_solution}'
                print(msg)

            min_index = np.argmin(error)
            if error[min_index] <= max_error:
                return old_population[min_index], error[min_index]

            if early_stopping_rounds != -1 and last_improvement > early_stopping_rounds:
                break

            if time_limit != -1 and time.time() - start_time > time_limit:
                break

            selector = selection(old_population, error, rng)
            new_population = []
            if elitism:
                new_population = [old_population[min_index]]

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
                    p1 = p_space[iteration]

                if eta == 'adaptive':
                    eta1 = eta_space[iteration]

                if adaptive_population:
                    pop_size1 = pop_size_space[iteration]

            elif not var_error0:
                var_error0 = var_error1

        return historical_best_solution, historical_min_error

    if isinstance(n_jobs, Parallel):
        return ga_(n_jobs)

    elif n_jobs == 1:
        return ga_(None)

    elif n_jobs == -1:
        with Parallel(n_jobs=os.cpu_count()) as parallel:
            return ga_(parallel)

    elif n_jobs > 1:
        with Parallel(n_jobs=n_jobs) as parallel:
            return ga_(parallel)

    else:
        raise ValueError(f'n_jobs must be an int >= 1 (got {n_jobs})')


ga = genetic_algorithm


def random_restarts(loss, guesses, search_func, *args,
                    rng=None,
                    verbose=False,
                    restarts=2,
                    **kwargs):
    """
    Minimize a loss function through an arbitrary search with
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


@search_algorithm
def mayfly_algorithm(loss, guesses,
                     a1=1,
                     a2=1.5,
                     B=2,
                     d=0.1,
                     fl=0.1,
                     client=None,
                     rng=None):

    guesses = np.array(guesses)
    rng.shuffle(guesses)
    sep = int(len(guesses) / 2)
    males = guesses[:sep]
    females = guesses[sep:]
    male_velocities = np.zeros((len(males),) + males[0].shape)
    female_velocities = np.zeros((len(females),) + females[0].shape)
    male_errors, female_errors = evaluate_solutions(loss, client, males, females)
    male_pbest = [m for m in zip(males, male_errors)]
    gbest_index = np.argmin(male_errors)
    gbest = males[gbest_index]
    gbest_error = male_errors[gbest_index]
    hbest, hbest_error = find_best((males, male_errors), (females, female_errors))
    yield 0, hbest, hbest_error, 'initialization'
    for iteration in infinite_count(1):
        male_velocities = update_male_velocities(males,
                                                 male_velocities,
                                                 male_pbest,
                                                 gbest,
                                                 a1,
                                                 a2,
                                                 B,
                                                 d,
                                                 rng)

        female_velocities = update_female_velocities(females,
                                                     female_velocities,
                                                     female_errors,
                                                     males,
                                                     male_errors,
                                                     fl,
                                                     a2,
                                                     B,
                                                     rng)

        males = males + male_velocities
        females = females + female_velocities
        male_errors, female_errors = evaluate_solutions(loss, client, males, females)
        gbest, gbest_error = find_best((males, male_errors),
                                       hbest=(gbest, gbest_error))

        hbest, hbest_error = find_best((males, male_errors),
                                       (females, female_errors),
                                       hbest=(hbest, hbest_error))

        yield iteration, hbest, hbest_error, 'evaluate males and females'
        sort_indices = np.argsort(male_errors)
        males = males[sort_indices]
        male_velocities = male_velocities[sort_indices]
        male_errors = male_errors[sort_indices]
        sort_indices = np.argsort(female_errors)
        females = females[sort_indices]
        female_velocities = female_velocities[sort_indices]
        female_errors = female_errors[sort_indices]
        offspring = mate_mayflies(males, male_errors, females, female_errors, rng)
        rng.shuffle(offspring)
        offspring_errors = evaluate_solutions(loss, client, offspring)
        hbest, hbest_error = find_best((offspring, offspring_errors),
                                       hbest=(hbest, hbest_error))

        yield iteration, hbest, hbest_error, 'evaluate offspring'
        male_offspring = offspring[:sep]
        female_offspring = offspring[sep:]
        male_offspring_errors = offspring_errors[:sep]
        female_offspring_errors = offspring_errors[sep:]
        males, male_velocities, male_errors, male_pbest = replace_worst(males,
                                                                        male_velocities,
                                                                        male_errors,
                                                                        male_offspring,
                                                                        male_offspring_errors,
                                                                        male_pbest)

        females, female_velocities, female_errors = replace_worst(females,
                                                                  female_velocities,
                                                                  female_errors,
                                                                  female_offspring,
                                                                  female_offspring_errors)

        male_pbest = update_pbest(males, male_errors, male_pbest)


ma = mayfly_algorithm


def with_cache(func):
    cache = dict()

    @wraps(func)
    def new_func(*args, **kwargs):
        return func(*args, cache=cache, **kwargs)

    return new_func


@with_cache
def evaluate_solutions(loss, client, solutions, *solution_groups, cache=None):
    solution_groups = (solutions,) + solution_groups
    futures = []
    for solutions in solution_groups:
        for solution in solutions:
            solution_hash = joblib_hash(solution)
            if solution_hash in cache:
                futures.append(cache[solution_hash])

            else:
                futures.append(client.submit(loss, solution))

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


def update_male_velocities(males, velocities, pbest, gbest, a1, a2, B, d, rng):
    pbest = np.array([p[0] for p in pbest])
    gbest = gbest[0]
    rp = np.sum(np.square(males - pbest), axis=1)
    rg = np.sum(np.square(males - gbest), axis=1)
    v2 = (velocities
          + (a1 * np.exp(-B * rp)).reshape((-1, 1)) * (pbest - males)
          + (a2 * np.exp(-B * rg)).reshape((-1, 1)) * (gbest - males))

    return (v2
            + d
            * rng.random(velocities.shape)
            * rng.choice([1, -1], size=velocities.shape))


def update_female_velocities(females, velocities, female_errors, males, male_errors, fl, a2, B, rng):
    rmf = np.sum(np.square(males - females), axis=1)
    return np.where((female_errors > male_errors).reshape((-1, 1)),
                    velocities + (a2 * np.exp(-B * rmf)).reshape((-1, 1)) * (males - females),
                    velocities + fl * rng.random(velocities.shape) * rng.choice([1, -1], size=velocities.shape))


def infinite_count(start=0):
    for i, _ in enumerate(repeat(None)):
        yield i + start


def find_best(solutions_errors, *solution_groups, hbest=None):
    if solution_groups:
        solutions, errors = list(zip(*solution_groups))
        solutions += (solutions_errors[0],)
        errors += (solutions_errors[1],)
        solutions = np.concatenate(solutions)
        errors = np.concatenate(errors)

    else:
        solutions, errors = solutions_errors

    if hbest:
        solutions = np.concatenate([solutions, [hbest[0]]])
        errors = np.concatenate([errors, [hbest[1]]])

    min_index = np.argmin(errors)

    return solutions[min_index], errors[min_index]


def mate_mayflies(males, male_errors, females, female_errors, rng):
    males = males[np.argsort(male_errors)]
    females = females[np.argsort(female_errors)]
    offspring = []
    for male, female in zip(males, females):
        offspring.extend(mayfly_crossover(male, female, rng))

    return np.array(offspring)


def mayfly_crossover(male, female, rng):
    length = int(round(rng.random() * len(male)))
    offspring1 = np.concatenate([male[:length], female[length:]])
    offspring2 = np.concatenate([female[:length], male[length:]])

    return offspring1, offspring2


def replace_worst(solutions,
                  velocities,
                  errors,
                  offspring,
                  offspring_errors,
                  pbest=None):
    pop_size = len(solutions)
    errors = np.concatenate([errors, offspring_errors])
    sort_array = np.argsort(errors)
    errors = errors[sort_array]
    solutions = np.concatenate([solutions, offspring])[sort_array]
    velocities = np.concatenate([velocities, np.zeros(offspring.shape)])[sort_array]
    solutions = solutions[:pop_size]
    velocities = velocities[:pop_size]
    errors = errors[:pop_size]
    if pbest:
        pbest = pbest + list(zip(offspring, offspring_errors))
        pbest = [p[1] for p in sorted(zip(sort_array, pbest))]
        pbest = pbest[:pop_size]
        return solutions, velocities, errors, pbest

    else:
        return solutions, velocities, errors


def update_pbest(males, male_errors, pbest_seq):
    new_pbest = []
    for solution, error, pbest in zip(males, male_errors, pbest_seq):
        if error < pbest[1]:
            new_pbest.append((solution, error))

        else:
            new_pbest.append(pbest)

    return new_pbest
