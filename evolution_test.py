from __future__ import print_function
import nest
import numpy as np
import time
nest.set_verbosity("M_ERROR")
np.set_printoptions(precision=4, suppress=True)
nest.SetKernelStatus({'grng_seed': int(time.time())})

max_generations = 3000
mutation_rate = 0.1
generation_size = 50
tournament_size = 5
crossover_rate = 0.5
elitism = False  # keep best individual

input = nest.Create("poisson_generator", 8)

params = {
    'V_th': float('inf'),
    'C_m': 10.0,
    'tau_m': 20.0,  # 20.0
    "tau_syn_in": 2.0,  # .5
    "tau_syn_ex": 2.0,  # .5
    "E_L": 0.0,
    "V_reset": 0.0,
    "t_ref": 0.1,
    "I_e": 0.0,
    "V_m": 0.0
}
nest.CopyModel("iaf_psc_alpha", "spike_sink", params)

params = {
    'tau_m': 20.0,
    'tau_syn_in': 2.0,  # 5.0
    'tau_syn_ex': 2.0,
    'I_e': 0.0,
    't_ref': 0.1,
    'V_th': -50.0,
    'V_reset': -65.0,
    'E_L': -65.0,
    'C_m': 1.0
}
nest.CopyModel("iaf_psc_exp", "exp_neuron", params)

output = nest.Create("exp_neuron", 3)
recorder = nest.Create("spike_detector")

initial_weights = np.random.rand(3, 8) * 5
nest.Connect(input, output, 'all_to_all', {"weight": initial_weights})
nest.Connect(output, recorder)
connections = nest.GetConnections(input, output)

ideal = np.zeros((8, 3))
for i in range(8):
    for x in range(3):
        ideal[i, x] = int(i & (2 ** x) > 0)


class Candidate(object):
    def __init__(self, weights):
        self.weights = weights
        self._error = None
        self.res = None

    @property
    def error(self):
        if self._error is None:
            self.calculate_fitness()
        return self._error

    def calculate_fitness(self):
        nest.SetStatus(connections, params="weight", val=self.weights)
        results = np.empty((8, 3))
        for i in range(8):
            nest.ResetNetwork()
            nest.SetStatus(input, params="rate", val=[500.0 if j == i else 0.0 for j in range(8)])
            nest.Simulate(100)
            events = nest.GetStatus(recorder, keys="events")[0]["senders"] - np.min(output)
            events = np.array([np.sum(events == j) for j in range(3)])
            results[i] = events
        maximum = np.max(results)
        if maximum != 0.0:
            results /= maximum
        self._error = np.mean(np.sum((results - ideal) ** 2, axis=1))
        self.res = results

    def cross(self, other):
        crossover = np.random.rand(len(self.weights)) < crossover_rate
        new_weights = self.weights.copy()
        new_weights[crossover] = other.weights[crossover]
        return Candidate(new_weights)

    def mutate(self):
        mutations = np.random.rand(len(self.weights)) < mutation_rate
        n_mutations = np.sum(mutations)
        if n_mutations > 0:
            new_values = np.random.rand(n_mutations) * 3.0
            self.weights[mutations] = new_values
            self._error = None
            self.res = None


# best_weights = np.array([0.0098, 0.5067, 1.1259, 2.0271, 0.1771, 1.6129, 0.5394, 2.5281,
#                          0.3598, 2.6433, 2.5404, 1.7086, 0.2211, 1.1071, 2.7847, 2.2118,
#                          1.1943, 2.3848, 0.304,  2.7886, 2.8018, 2.3895, 2.8755, 2.9001])
# best = Candidate(best_weights)
# scores = list()
# for _ in range(20):
#     scores.append(best.error)
#     best._error = None
# print("{:.3f}, {:.3f}".format(max(scores), min(scores)))
# print(best.res)
# print(np.concatenate((np.array(connections)[:, :2], np.array(nest.GetStatus(connections, keys="weight")).reshape(-1, 1)), axis=1))
# import sys; sys.exit(0)


def tournament_select(population):
    tournament = np.random.choice(population, tournament_size)
    return min(tournament, key=lambda x: x.error)


population = [Candidate(np.random.rand(8 * 3) * 5) for i in range(generation_size)]
best = min(population, key=lambda x: x.error)

generation = 0
try:
    while best.error > 0.001 and generation < max_generations:
        new_population = list()
        num = generation_size if not elitism else generation_size - 1
        if elitism:
            new_population.append(best)

        # odds = np.array([1.0 / (1.0 + x.error * 10) for x in population])
        # odds /= odds.sum()

        for i in range(num):
            ind1 = tournament_select(population)
            ind2 = tournament_select(population)
            # ind1, ind2 = np.random.choice(population, 2, False, p=odds)
            new_individual = ind1.cross(ind2)
            new_individual.mutate()
            new_population.append(new_individual)

        population = new_population
        best = min(population, key=lambda x: x.error)
        print("Generation {:04d}, best error is {:.3f}, result {}".format(generation, best.error, best.res[7]), end='\r')
        generation += 1
except KeyboardInterrupt:
    pass
finally:
    print()
    best = min(population, key=lambda x: x.error)
    print(best.weights)
