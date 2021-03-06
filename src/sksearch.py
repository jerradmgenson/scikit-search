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

import math

import numpy as np

import mayflies
import searchlib as sl


@sl.search_algorithm
def particle_swarm_optimization(loss, guesses,
                                c1=2,
                                c2=2,
                                vmax=2,
                                max_iter=1000,
                                n_jobs=1,
                                client=None,
                                rng=None):
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
      max_time: Amount of time that genetic algorithm is allowed to
                run in seconds. `-1` means no time limit. Default is `-1`.
      n_jobs: Number of processes to use when evaluating the loss function `-1`
              creates a process for each available CPU. May also be an instance
              of `joblib.Parallel`. Default is `1`.
      client: An instance of `dask.distributed.Client`. Default is a new Client
              that uses local CPUs for concurrency.
      rng: An instance of numpy.random.Generator. If not given, a new Generator
           will be created.
      verbose: Set to `True` to print the error on each iteration. Default
               is `False`.

    Returns:
      A tuple of (best_solution, error).

    """

    historical_best_solution = None
    historical_min_error = np.inf
    guesses = np.array(guesses)
    pbest = guesses
    pbest_error = np.full(len(guesses), np.inf)
    velocity = np.zeros((len(guesses),) + guesses[0].shape)
    for iteration in sl.infinite_count():
        gbest = None
        gbest_error = np.inf
        error = sl.evaluate_solutions(loss, client, guesses)
        min_index = np.argmin(error)
        gbest_error = error[min_index]
        gbest = guesses[min_index]
        if gbest_error < historical_min_error:
            historical_best_solution = gbest
            historical_min_error = gbest_error

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
        yield iteration, historical_best_solution, historical_min_error, ''


pso = particle_swarm_optimization


@sl.search_algorithm
def genetic_algorithm(loss, guesses,
                      p='auto',
                      eta='auto',
                      adaptive_population=False,
                      elitism=False,
                      max_iter=1000,
                      n_jobs=1,
                      client=None,
                      rng=None,
                      crossover=sl.uniform_crossover,
                      mutate=sl.default_mutate,
                      selection=sl.fitness_proportional_selection):
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
      client: An instance of `dask.distributed.Client`. Default is a new Client
              that uses local CPUs for concurrency.
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
    for iteration in sl.infinite_count():
        error = sl.evaluate_solutions(loss, client, old_population)
        iteration_min_error = np.min(error)
        if iteration_min_error < historical_min_error:
            historical_min_error = iteration_min_error
            historical_best_solution = old_population[np.argmin(error)]

        yield iteration, historical_best_solution, historical_min_error, ''
        min_index = np.argmin(error)
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


ga = genetic_algorithm


@sl.search_algorithm
def mayfly_algorithm(loss, guesses,
                     a1=1,
                     a2=1.5,
                     beta=2,
                     d=0.1,
                     fl=0.1,
                     sigma='adaptive',
                     vmax=None,
                     gmax=2.5,
                     p='adaptive',
                     eta='adaptive',
                     max_iter=None,
                     client=None,
                     rng=None):
    """
    Minimize a loss function using mayfly optimization algorithm.

    Args:
      loss: The loss function to be minimized. Accepts objects of the
            same type as guesses and returns a 1-D ndarray of error scores,
            where lower scores are better.
      guesses: A 2-D array-like object containing candidate solutions to the
               search problem. Should be compatible with numpy.ndarray.
      a1: positive attraction constant used to scale the contribution of
          the cognitive component. Default is 1.
      a2: positive attraction constant used to scale the contribution of
          the social component. Default is 1.5.
      beta: fixed visibility coefficient used to limit a mayfly's visibility
            to others. Default is 2.
      d: nupital dance coefficient in range (0, 1). Default is 0.1.
      fl: random walk coefficient in range (0, 1). . Default is 0.1.
      sigma: reduction coefficient for d and fl in range (0, 1).
             Can also be 'adaptive', in which case sigma is calculated as
             (max_iter - curr_iter) / max_iter. Default is 'adaptive'.
      vmax: maximum velocity constraint. May be either a float, an array-like
            object, 'auto', or None. If set to 'auto', vmax will be calculated
            as `rand * (xmax - xmin). If set to None, vmax will not be checked.
            Default is None.
      gmax: maximum value for the gravity coefficient. May be a float or None.
            If None, gravity coefficient will not be used. If a float, the
            gravity coefficient will be reduced on each iteration.
            Default is 2.
      p: Controls the frequency of mutations, i.e. the probability that each
         element in a "child" solution will be mutated. May be a float, 'auto',
          or 'adaptive'. If set to 'auto', a heuristic is used to set a value
          for `p`. If set to 'adaptive', a heurstic is used to set an initial
          value and shrinking is applied as the number of iterations increase.
      eta: Controls the magnitude of mutations, where higher values of eta
           correspond to higher magnitudes. May be a float, 'auto', or
           'adaptive'. If set to 'auto', a heuristic is used to set a value for
           `eta`. If set to 'adaptive', a heurstic is used to set an initial
           value and shrinking is applied as the number of iterations increase.
      max_error: Maximum error score required for early stopping. Defaults to
                 `0`.
      max_iter: Maximum number of iterations before the function returns.
                Defaults to `1000`.
      early_stopping_rounds: The number of iterations that are allowed to pass
                             without improvement before the function returns.
                             `-1` indicates no early stopping. Default is `-1`.
      max_time: Amount of time that genetic algorithm is allowed to
                run in seconds. `-1` means no time limit. Default is `-1`.
      n_jobs: Number of processes to use when evaluating the loss function `-1`
              creates a process for each available CPU. May also be an instance
              of `joblib.Parallel`. Default is `1`.
      client: An instance of `dask.distributed.Client`. Default is a new Client
              that uses local CPUs for concurrency.
      rng: An instance of numpy.random.Generator. If not given, a new Generator
           will be created.
      verbose: Set to `True` to print the error on each iteration. Default
               is `False`.

    Returns:
      A tuple of (best_solution, error).

    References:
      - Zervoudakis, K. & Tsafarakis, S. (2020). A mayfly optimization algorithm.

    """

    guesses = np.array(guesses)
    if vmax == 'auto':
        vmax = rng.random((1, guesses.shape[1])) * (np.max(guesses, axis=0) - np.min(guesses, axis=0))

    elif vmax is not None:
        vmax = np.array(vmax).reshape((1, -1))

    # Separate guesses into male and female mayflies.
    rng.shuffle(guesses)
    sep = int(len(guesses) / 2)
    males = guesses[:sep]
    females = guesses[sep:]

    # Initialize mayfly velocities.
    male_velocities = np.zeros((len(males),) + males[0].shape)
    female_velocities = np.zeros((len(females),) + females[0].shape)

    # Calculate initial errors.
    male_errors, female_errors = sl.evaluate_solutions(loss, client, males, females)

    # Calculate personal best (pbest), global best (gbest), and
    # historical best (hbest).
    male_pbest = [m for m in zip(males, male_errors)]
    gbest_index = np.argmin(male_errors)
    gbest = males[gbest_index]
    gbest_error = male_errors[gbest_index]
    hbest, hbest_error = sl.find_best((males, male_errors), (females, female_errors))

    # Initialize gravity coefficients
    g = gmax
    if g is not None:
        gmin = gmax * 0.1

    # Initialize sigma coefficients
    sigma0 = sigma

    # Initialize eta
    if eta == 'auto':
        eta_upper = np.std(guesses, axis=0)
        eta_lower = eta_upper * 0.1
        eta0 = np.geomspace(eta_upper, eta_lower, max_iter)[int(max_iter / 2)]

    elif eta == 'adaptive':
        eta0 = np.std(guesses, axis=0)
        eta_min = eta0 * 0.1
        eta_space = np.linspace(eta0, eta_min, max_iter)

    else:
        eta0 = eta

    # Initialize p
    if p == 'auto':
        p0 = 1 / guesses.shape[1]

    elif p == 'adaptive':
        p0 = 1
        p_min = 1 / guesses.shape[1] / 2
        p_space = np.linspace(p0, p_min, max_iter)

    else:
        p0 = p

    p1 = p0
    eta1 = eta0

    yield 0, hbest, hbest_error, 'initialization'
    for iteration in sl.infinite_count(1):
        male_velocities = mayflies.update_male_velocities(males,
                                                          male_velocities,
                                                          male_pbest,
                                                          gbest,
                                                          a1,
                                                          a2,
                                                          beta,
                                                          d,
                                                          vmax,
                                                          g,
                                                          rng)

        if sigma0 == 'adaptive':
            sigma = (max_iter - iteration + 1) / max_iter

        d = d * sigma
        female_velocities = mayflies.update_female_velocities(females,
                                                              female_velocities,
                                                              female_errors,
                                                              males,
                                                              male_errors,
                                                              fl,
                                                              a2,
                                                              beta,
                                                              vmax,
                                                              g,
                                                              rng)

        fl = fl * sigma
        if g is not None:
            # Reduce gravity coefficient.
            g = gmax - (gmax - gmin) / max_iter * iteration

        # Update positions.
        males = males + male_velocities
        females = females + female_velocities

        male_errors, female_errors = sl.evaluate_solutions(loss, client, males, females)
        gbest, gbest_error = sl.find_best((males, male_errors),
                                          hbest=(gbest, gbest_error))

        hbest, hbest_error = sl.find_best((males, male_errors),
                                          (females, female_errors),
                                          hbest=(hbest, hbest_error))

        yield iteration, hbest, hbest_error, 'evaluate parents'

        # Sort male and females mayflies according to their errors.
        sort_indices = np.argsort(male_errors)
        males = males[sort_indices]
        male_velocities = male_velocities[sort_indices]
        male_errors = male_errors[sort_indices]
        sort_indices = np.argsort(female_errors)
        females = females[sort_indices]
        female_velocities = female_velocities[sort_indices]
        female_errors = female_errors[sort_indices]

        offspring = mayflies.breed_mayflies(males,
                                            male_errors,
                                            females,
                                            female_errors,
                                            rng)

        offspring = np.array([sl.default_mutate(a, p1, eta1, rng) for a in offspring])
        if p == 'adaptive':
            p1 = p_space[iteration-1]

        if eta == 'adaptive':
            eta1 = eta_space[iteration-1]

        rng.shuffle(offspring)
        offspring_errors = sl.evaluate_solutions(loss, client, offspring)
        hbest, hbest_error = sl.find_best((offspring, offspring_errors),
                                          hbest=(hbest, hbest_error))

        yield iteration, hbest, hbest_error, 'evaluate offspring'

        # Separate offspring into male and female mayflies.
        male_offspring = offspring[:sep]
        female_offspring = offspring[sep:]
        male_offspring_errors = offspring_errors[:sep]
        female_offspring_errors = offspring_errors[sep:]

        males, male_velocities, male_errors, male_pbest = mayflies.replace_worst(males,
                                                                                 male_velocities,
                                                                                 male_errors,
                                                                                 male_offspring,
                                                                                 male_offspring_errors,
                                                                                 male_pbest)

        females, female_velocities, female_errors = mayflies.replace_worst(females,
                                                                           female_velocities,
                                                                           female_errors,
                                                                           female_offspring,
                                                                           female_offspring_errors)

        male_pbest = mayflies.update_pbest(males, male_errors, male_pbest)


ma = mayfly_algorithm


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
