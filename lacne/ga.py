"""Genetic-algorithm baseline for the upper-level mixed-integer problem."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GAResult:
    z: np.ndarray
    c: np.ndarray
    f: float
    all_f: np.ndarray
    best_so_far: np.ndarray


class GA:
    def __init__(self, evaluate, d_z: int, n_select: int,
                 c_min: float, c_max: float,
                 pop_size: int = 20, budget: int = 75, seed: int = 0,
                 mutation_rate: float = 0.2) -> None:
        self.evaluate = evaluate
        self.d_z = d_z
        self.n_select = n_select
        self.c_min = c_min
        self.c_max = c_max
        self.pop_size = pop_size
        self.budget = budget
        self.seed = seed
        self.mutation_rate = mutation_rate

    def _random_z(self, rng) -> np.ndarray:
        z = np.zeros(self.d_z)
        z[rng.choice(self.d_z, self.n_select, replace=False)] = 1.0
        return z

    def _repair_z(self, z, rng) -> np.ndarray:
        z = z.copy()
        k = int(z.sum())
        while k < self.n_select:
            idx = np.where(z == 0)[0]
            z[rng.choice(idx)] = 1.0
            k += 1
        while k > self.n_select:
            idx = np.where(z == 1)[0]
            z[rng.choice(idx)] = 0.0
            k -= 1
        return z

    def _crossover(self, p1, p2, rng):
        z1, c1 = p1
        z2, c2 = p2
        mask = rng.random(self.d_z) < 0.5
        child_z = np.where(mask, z1, z2)
        child_z = self._repair_z(child_z, rng)
        alpha = rng.random(self.d_z)
        child_c = alpha * c1 + (1.0 - alpha) * c2
        child_c = np.clip(child_c, self.c_min, self.c_max)
        return child_z, child_c

    def _mutate(self, ind, rng):
        z, c = ind
        z = z.copy(); c = c.copy()
        if rng.random() < self.mutation_rate and self.d_z >= 2:
            ones = np.where(z == 1)[0]
            zeros = np.where(z == 0)[0]
            if len(ones) and len(zeros):
                i = rng.choice(ones); j = rng.choice(zeros)
                z[i], z[j] = 0.0, 1.0
        c = c + rng.normal(0.0, 0.1 * (self.c_max - self.c_min), self.d_z)
        c = np.clip(c, self.c_min, self.c_max)
        return z, c

    def run(self) -> GAResult:
        rng = np.random.default_rng(self.seed)
        pop = []
        fvals = []
        all_f = []
        best_so_far = []
        for _ in range(self.pop_size):
            z = self._random_z(rng)
            c = rng.uniform(self.c_min, self.c_max, self.d_z)
            f = float(self.evaluate(z, c))
            pop.append((z, c)); fvals.append(f)
            all_f.append(f)
            best_so_far.append(min(best_so_far[-1], f) if best_so_far else f)
        fvals = np.asarray(fvals)

        n_evals = self.pop_size
        while n_evals < self.budget:
            # roulette-wheel (fitness-proportionate) selection
            fmax = fvals.max()
            fit = (fmax - fvals) + 1e-9
            p = fit / fit.sum()
            parent_idx = rng.choice(self.pop_size, size=self.pop_size, p=p)
            parents = [pop[int(i)] for i in parent_idx]
            # crossover + mutation: produce two children per parent pair
            offspring = []
            of = []
            for i in range(0, self.pop_size, 2):
                j = (i + 1) % self.pop_size
                for _ in range(2):
                    cz, cc = self._crossover(parents[i], parents[j], rng)
                    cz, cc = self._mutate((cz, cc), rng)
                    f = float(self.evaluate(cz, cc))
                    offspring.append((cz, cc)); of.append(f)
                    n_evals += 1
                    all_f.append(f)
                    best_so_far.append(min(best_so_far[-1], f))
                    if n_evals >= self.budget:
                        break
                if n_evals >= self.budget:
                    break
            # If the budget was exhausted mid-generation, stop here (the
            # partial offspring are already recorded in best_so_far).
            if n_evals >= self.budget and len(offspring) < self.pop_size:
                break

            # basic generational GA: keep the single best parent (elitism of 1)
            elite_k = min(1, self.pop_size)
            order = np.argsort(fvals)[:elite_k]
            new_pop = [pop[i] for i in order]
            new_f = [fvals[i] for i in order]
            of_arr = np.asarray(of)
            of_order = np.argsort(of_arr)[:(self.pop_size - elite_k)]
            for i in of_order:
                new_pop.append(offspring[int(i)])
                new_f.append(of_arr[int(i)])
            pop = new_pop
            fvals = np.asarray(new_f)

        best_i = int(np.argmin(fvals))
        z, c = pop[best_i]
        return GAResult(z=z, c=c, f=float(fvals[best_i]),
                        all_f=np.asarray(all_f), best_so_far=np.asarray(best_so_far))
