"""
A library of functions related to mayfly optimization algorithm.

Copyright 2021 Jerrad Michael Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

import numpy as np


def update_male_velocities(males, velocities, pbest, gbest, a1, a2, beta, d, vmax, g, rng):
    """
    Update male velocities for mayfly algorithm.

    Returns:
      A new 2-D array of velocities.

    """

    pbest = np.array([p[0] for p in pbest])
    gbest = gbest[0]
    rp = np.sum(np.square(males - pbest), axis=1)
    rg = np.sum(np.square(males - gbest), axis=1)
    if g is not None:
        velocities = g * velocities

    v2 = (velocities
          + (a1 * np.exp(-beta * rp)).reshape((-1, 1)) * (pbest - males)
          + (a2 * np.exp(-beta * rg)).reshape((-1, 1)) * (gbest - males))

    velocities = (v2
                  + d
                  * rng.random(velocities.shape)
                  * rng.choice([1, -1], size=velocities.shape))

    velocities = check_vmax(velocities, vmax)

    return velocities


def update_female_velocities(females, velocities, female_errors, males, male_errors, fl, a2, beta, vmax, g, rng):
    """
    Update female velocities for mayfly algorithm.

    Returns:
      A new 2-D array of velocities.

    """

    rmf = np.sum(np.square(males - females), axis=1)
    if g is not None:
        velocities = g * velocities

    eq1 = velocities + (a2 * np.exp(-beta * rmf)).reshape((-1, 1)) * (males - females)
    eq2 = velocities + fl * rng.random(velocities.shape) * rng.choice([1, -1], size=velocities.shape)
    velocities = np.where((female_errors > male_errors).reshape((-1, 1)),
                          eq1,
                          eq2)

    velocities = check_vmax(velocities, vmax)

    return velocities


def check_vmax(velocities, vmax):
    """
    Check that the values in `velocities` do not exceed (+-) vmax.

    """

    if vmax is None:
        return velocities

    velocities = np.where(velocities < vmax,
                          velocities,
                          vmax)

    velocities = np.where(velocities > -vmax,
                          velocities,
                          -vmax)

    return velocities


def breed_mayflies(males, male_errors, females, female_errors, rng):
    """
    Mating function for mayfly algorithm.

    Returns:
      A 2-D array of offspring mayflies.

    """

    males = males[np.argsort(male_errors)]
    females = females[np.argsort(female_errors)]
    offspring = []
    for male, female in zip(males, females):
        offspring.extend(mayfly_crossover(male, female, rng))

    offspring = np.array(offspring)

    return offspring


def mayfly_crossover(male, female, rng):
    """
    Crossover operator for mayfly algorithm.

    Returns:
      A tuple of (offspring1, offspring2).

    """

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
    """
    Replace solutions in `solutions` with better solutions from `offspring`.

    Returns:
      A tuple of (solutions, velocities, errors) with the len(solutions) best
      solutions from `solutions` and `offspring`. If `pbest` is given, then
      the returned tuple will be (solutions, velocities, errors, pbest).

    """

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
    """
    Update pbest values for male mayflies.

    """

    new_pbest = []
    for solution, error, pbest in zip(males, male_errors, pbest_seq):
        if error < pbest[1]:
            new_pbest.append((solution, error))

        else:
            new_pbest.append(pbest)

    return new_pbest
