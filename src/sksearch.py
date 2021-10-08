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

import enum

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


def pso(solutions, loss,
        c1=2,
        c2=2,
        vmax=2,
        min_error=0,
        max_iter=1000,
        rng=None):
    if not rng:
        rng = np.random.default_rng()

    solutions = list(solutions)
    pbest = solutions
    pbest_fitness = list(np.full(len(solutions), np.inf))
    velocity = list(np.full((len(solutions),) + solutions[0].shape, 0))
    for _ in range(max_iter):
        gbest = None
        gbest_fitness = np.inf
        for i, solution in enumerate(solutions):
            fitness = loss(solution)
            if fitness <= min_error:
                return solution, fitness

            if fitness < pbest_fitness[i]:
                pbest_fitness[i] = fitness
                pbest[i] = solution

            if gbest is None or fitness < gbest_fitness:
                gbest = solution
                gbest_fitness = fitness

        for i, solution in enumerate(solutions):
            velocity[i] = (velocity[i]
                           + c1
                           * rng.random()
                           * (pbest[i] - solution)
                           + c2
                           * rng.random()
                           * (gbest - solution))

            velocity[i] = np.where(np.abs(velocity[i]) > vmax,
                                   vmax * np.sign(velocity[i]),
                                   velocity[i])

        solutions = [s + velocity[i] for i, s in enumerate(solutions)]

    return gbest, gbest_fitness
