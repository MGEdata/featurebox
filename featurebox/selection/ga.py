import random
from collections import deque
from functools import partial

import numpy as np
from deap import base
from deap import tools
from deap.algorithms import varAnd
from deap.tools import mutShuffleIndexes
from mgetool.newclass import create
from mgetool.tool import check_random_state, parallelize
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.feature_selection import SelectorMixin
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection._search import BaseSearchCV
from sklearn.utils.validation import check_is_fitted

from featurebox.selection.multibase import MultiBase


def eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None, n_jobs=2,
             halloffame=None, verbose=__debug__):
    """This algorithm reproduce the simplest evolutionary algorithm.

    :param population: A list of individuals.
    :param n_jobs: jobs.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution

    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    # n_job=n
    invalid_ind2 = [tuple(i) for i in invalid_ind]
    # fitnesses = batch_parallelize(n_jobs, toolbox.evaluate, invalid_ind2, batch_size=30, tq=True)
    fitnesses = parallelize(n_jobs, toolbox.evaluate, invalid_ind2, tq=True)
    # n_job=1
    # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    best = deque([halloffame.items[0].fitness.values[0]], maxlen=15)

    # Begin the generational process
    for gen in range(1, ngen + 1):

        # Append the current generation statistics to the logbook
        record = stats.compile(population + halloffame.items) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - halloffame.maxsize)
        offspring.extend(halloffame.items)
        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)
        offspring = toolbox.map(toolbox.filt, offspring)
        offspring = list(offspring)
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        # n_job=n
        invalid_ind2 = [tuple(i) for i in invalid_ind]

        # fitnesses = batch_parallelize(n_jobs, toolbox.evaluate, invalid_ind2, batch_size=30, tq=True)
        fitnesses = parallelize(n_jobs, toolbox.evaluate, invalid_ind2, tq=True)
        # n_job=1
        # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        best.append(halloffame.items[0].fitness.values[0])
        if sum(best) / best.maxlen == best[-1]:
            break
        # Replace the current population by the offspring
        population[:] = offspring

    return population, logbook


def generate_xi():
    return random.randint(0, 1)


def generate(space):
    return [generate_xi() for _ in range(space)]


def filt(ind, min_=2, max_=None):
    if max_ is not None:
        if np.sum(ind) > max_:
            index = np.where(np.array(ind) == 1)[0]
            k = random.randint(min_, max_)
            index2 = random.sample(list(index), k=k)
            ind[:] = [0] * len(ind)
            [ind.__setitem__(i, 1) for i in index2]

        elif np.sum(ind) < min_:
            k = random.randint(min_, max_)
            index2 = random.sample(list(range(len(ind))), k=k)
            ind[:] = [0] * len(ind)
            [ind.__setitem__(i, 1) for i in index2]

    else:
        if np.sum(ind) < min_:
            k = random.randint(min_, len(ind))
            index2 = random.sample(list(range(len(ind))), k=k)
            ind[:] = [0] * len(ind)
            [ind.__setitem__(i, 1) for i in index2]

    # if np.sum(ind)<min_:
    #     raise UserWarning("???")

    return ind


class GA(BaseEstimator, MetaEstimatorMixin, SelectorMixin, MultiBase):
    """
    GA with binding. Please just passing training data.

    Examples
    ---------
    >>> from sklearn.datasets import fetch_california_housing
    >>> from sklearn.svm import SVR
    >>> data = fetch_california_housing()
    >>> x = data.data[:50]
    >>> y = data.target[:50]
    >>> svr = SVR(gamma="scale", C=100)
    >>> ga = GA(estimator=svr, n_jobs=2, pop_n=50, hof_n=1, cxpb=0.8, mutpb=0.4, ngen=3, max_or_min="max", mut_indpb=0.1, min_=2, multi_index=[0, 5],random_state=0)
    >>> ga.fit(x_rain, y_train)

    Then

    >>> ga.score(x_test, y_test)

    """

    def __init__(self, estimator, n_jobs=2, pop_n=1000, hof_n=1, cxpb=0.6, mutpb=0.3, ngen=40, max_or_min="max",
                 mut_indpb=0.05, max_=None, min_=2, random_state=None, multi_grade=2, multi_index=None, must_index=None,
                 cv: int = 5, scoring=None):
        """

        Parameters
        ----------
        estimator:
            sklearn estimator
        n_jobs:int
            njobs
        pop_n:int
            population
        hof_n:int
            hof
        cxpb:float
            probility of cross
        mutpb:float
            probility of mutate
        ngen:int
            generation
        max_or_min:str
            "max","min";max problem or min
        mut_indpb:float
            probility of mutate of each node.
        max_:int
            max size
        min_:int
            min size
        random_state:float
            randomstate
        multi_grade:
            binding grade
        multi_index:
            binding range [min,max]
        scoring:None,str
            scoring method name.
        cv:bool
            if estimator is sklearn model, used cv, else pass.
        """
        super().__init__(multi_grade=multi_grade, multi_index=multi_index, must_index=must_index)
        assert cv >= 3
        if isinstance(estimator, BaseSearchCV):
            print(f"Using scoring:{scoring},and cv:{cv}")
            estimator.scoring = scoring
            self.cv = cv
        self.scoring = scoring
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.pop_n = pop_n
        self.hof_n = hof_n
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.ngen = ngen
        self.mut_indpb = mut_indpb
        self.max_ = max_
        self.min_ = min_
        self.max_or_min = max_or_min
        self.random_state = random_state
        self.cv = cv

        check_random_state(random_state)
        random.seed(random_state)
        np.random.seed(0)

        self.toolbox = base.Toolbox()
        if max_or_min == "max":
            FitnessMax = create("FitnessMax", base.Fitness, weights=(1.0,))
        else:
            FitnessMax = create("FitnessMax", base.Fitness, weights=(-1.0,))
        self.Individual = create("Individual", list, fitness=FitnessMax)
        self.toolbox = base.Toolbox()
        # Attribute generator

    @staticmethod
    def generate_min_max(space, min_=2, max_=None):
        ind = [generate_xi() for _ in range(space)]
        return filt(ind, min_=min_, max_=max_)

    def feature_fold_length(self, feature):

        multi_grade, multi_index = self.multi_grade, self.multi_index
        if self.check_multi:
            cc = []
            feature = np.sort(feature)

            i = 0
            while i <= feature[-1]:
                if multi_index[0] <= i < multi_index[1]:
                    i += self.multi_grade
                    cc.append(self.multi_grade)
                else:
                    i += 1
                    cc.append(1)
            return cc
        else:
            return [1] * len(feature)

    def fit(self, X, y):
        """Fit data and run GA."""
        x_space = X.shape[1]
        self.X = X
        self.y = y
        self.x_space_fold = self.feature_fold_length(list(range(x_space)))
        toolbox = self.toolbox
        toolbox.register("generate_x", self.generate_min_max, len(self.x_space_fold), min_=self.min_, max_=self.max_)
        # Structure initializers
        toolbox.register("individual", tools.initIterate, self.Individual,
                         toolbox.generate_x)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        fit_func = partial(self.fitness_func, model=self.estimator, x=X, y=y, return_model=False)
        toolbox.register("evaluate", fit_func)
        toolbox.register("filt", filt, min_=self.min_, max_=self.max_)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", mutShuffleIndexes, indpb=self.mut_indpb)
        toolbox.register("select", tools.selTournament, tournsize=3)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop = toolbox.population(n=self.pop_n)
        self.hof = tools.HallOfFame(self.hof_n)

        eaSimple(pop, self.toolbox, cxpb=self.cxpb, mutpb=self.mutpb, ngen=self.ngen, n_jobs=self.n_jobs,
                 stats=stats, halloffame=self.hof, verbose=True)
        for i in self.hof.items:
            print(self.unfold(i), i.fitness)
        support_ = self.unfold(self.hof.items[0])
        self.support_ = np.array(support_) > 0
        return self

    def unfold(self, ind):
        sss = []
        [sss.extend([i] * j) for i, j in zip(ind, self.x_space_fold)]
        if self.must_index is not None:
            [sss.__setitem__(i, 1) for i in self.must_index]
        return sss

    def fitness_func(self, ind, model, x, y, return_model=False):
        sss = self.unfold(ind)
        index = np.where(np.array(sss) == 1)[0]
        x = x[:, index]
        if x.shape[1] > 1:
            svr = model
            if hasattr(svr, "max_features"):
                svr.max_features = x.shape[1]
            svr.fit(x, y)
            if isinstance(svr, BaseSearchCV):
                sc = svr.best_score_
            else:
                sc = np.mean(cross_val_score(svr, x, y, cv=self.cv, scoring=self.scoring))
            if return_model:
                return sc, svr
            else:
                return sc,
        else:
            if return_model:
                return 0, None
            else:
                return 0,

    def socre_func(self, ind, model, x, y):
        sss = self.unfold(ind)
        index = np.where(np.array(sss) == 1)[0]
        x = x[:, index]
        if x.shape[1] > 1:
            svr = model
            y2 = svr.predict(x)
            sc = r2_score(y, y2)
            return sc
        else:
            raise TypeError("only one feature, error")

    def predict_func(self, ind, model, x):
        sss = self.unfold(ind)
        index = np.where(np.array(sss) == 1)[0]
        x = x[:, index]
        if x.shape[1] > 1:
            svr = model
            y = svr.predict(x)
            return y
        else:
            raise TypeError("only one feature,error")

    def _get_support_mask(self):
        check_is_fitted(self, 'support_')
        return self.support_

    def score_cv(self, X, y):
        """Reduce X to the selected feature and then return the score of the underlying estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_feature]
            The input0 samples.

        y : array of shape [n_samples]
            The target values.
        """

        mod = self.fitness_func(self.hof.items[0], self.estimator, self.X, self.y, return_model=True)[1]
        score = self.fitness_func(self.hof.items[0], mod, X, y, return_model=False)
        return score

    def score(self, X, y):
        """Reduce X to the selected feature and then return the score of the underlying estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_feature]
            The input0 samples.

        y : array of shape [n_samples]
            The target values.
        """

        mod = self.fitness_func(self.hof.items[0], self.estimator, self.X, self.y, return_model=True)[1]
        score = self.socre_func(self.hof.items[0], mod, X, y)
        return score

    def predict(self, X):
        """Reduce X to the selected feature and then return the score of the underlying estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_feature]
            The input0 samples.
        """
        mod = self.fitness_func(self.hof.items[0], self.estimator, self.X, self.y, return_model=True)[1]
        score = self.predict_func(self.hof.items[0], mod, X)
        return score

# if __name__ == "__main__":
#     from sklearn.svm import SVR
#     from sklearn.datasets import fetch_california_housing
#     data = fetch_california_housing()
#     x = data.data
#     y = data.target
#     x = x[:100]
#     y = y[:100]
#     svr = SVR(gamma="scale", C=100)
#
#     ga = GA(estimator=svr, n_jobs=2, pop_n=100, hof_n=1, cxpb=0.8, mutpb=0.4, ngen=10,
#             max_or_min="max", mut_indpb=0.1, min_=2, multi_index=[0, 5], random_state=0)
#     ga.fit(x, y)
#     ga.score(x, y)
