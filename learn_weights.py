# disable global logging from the virtual coach
import logging
import time
import numpy as np
import rospy
from std_msgs.msg import Float32
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# logging.disable(logging.INFO)
logging.getLogger('rospy').propagate = False
logging.getLogger('rosout').propagate = False
np.set_printoptions(precision=4, suppress=True)

max_generations = 100
mutation_rate = 0.1
generation_size = 20
tournament_size = 5
crossover_rate = 0.5
elitism = False  # keep best individual

brain_template = \
'''
from hbp_nrp_cle.brainsim import simulator as sim
from numpy import array
sensors = sim.Population(3, cellclass=sim.IF_curr_exp())
actors = sim.Population(6, cellclass=sim.IF_curr_exp())

projection = sim.Projection(sensors, actors, sim.AllToAllConnector(), sim.StaticSynapse(weight={syn_weight}))

circuit = sensors + actors
'''

distance_topic = "ball_distance"
candidate_index = 0
last_restart = None

class Candidate(object):
    def __init__(self, weights, parents=None):
        global candidate_index
        self.index = candidate_index
        candidate_index += 1
        self.weights = weights
        self._fitness = None
        self.parents = parents

    @property
    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    def calculate_fitness(self):
        brain = brain_template.format(syn_weight=repr(self.weights.reshape(3, 6)))
        global last_restart
        if (time.time() - last_restart) > 50 * 60:
            global sim
            sim.stop()
            time.sleep(5)
            sim = vc.launch_experiment('ExDHBPPrak_Motion')
            last_restart = time.time()
            sim._Simulation__logger.setLevel(logging.WARNING)
        sim.edit_brain(brain)
        sim.start()
        resultA = rospy.wait_for_message(distance_topic, Float32).data
        resultB = rospy.wait_for_message(distance_topic, Float32).data
        sim.pause()
        self._fitness = max(resultA, resultB)
        logger.debug("Distance for {}: {:.2f}m".format(self, self._fitness))

    def cross(self, other):
        crossover = np.random.rand(len(self.weights)) < crossover_rate
        new_weights = self.weights.copy()
        new_weights[crossover] = other.weights[crossover]
        return Candidate(new_weights, (self.index, other.index))

    def mutate(self):
        mutations = np.random.rand(len(self.weights)) < mutation_rate
        n_mutations = np.sum(mutations)
        if n_mutations > 0:
            new_values = np.random.rand(n_mutations) * 5.0
            self.weights[mutations] = new_values
            self._error = None
            self.res = None

    def __str__(self):
        if self.parents is None:
            return str(self.index)
        else:
            return "{} {}".format(self.index, self.parents)


def tournament_select(population):
    tournament = np.random.choice(population, tournament_size)
    return max(tournament, key=lambda x: x.fitness)

def genetic_search():
    population = [Candidate(np.random.rand(3 * 6) * 5.0) for i in range(generation_size)]

    try:
        for generation in range(max_generations):
            best = max(population, key=lambda x: x.fitness)
            logger.info("Generation {:03d}, best distance is {:.2f}m".format(generation, best.fitness))
            logger.info("{} {}".format(best, repr(best.weights)))

            new_population = list()
            num = generation_size if not elitism else generation_size - 1
            if elitism:
                new_population.append(best)

            for i in range(num):
                ind1 = tournament_select(population)
                ind2 = tournament_select(population)

                new_individual = ind1.cross(ind2)
                new_individual.mutate()
                new_population.append(new_individual)

            population = new_population
    except (KeyboardInterrupt, rospy.exceptions.ROSInterruptException):
        pass
    finally:
        best = max(population, key=lambda x: x._fitness)
        logger.critical(best._fitness)
        logger.critical(repr(best.weights))

# log into the virtual coach, update with your credentials
try:
    from hbp_nrp_virtual_coach.virtual_coach import VirtualCoach
    vc = VirtualCoach(environment='local')
except ImportError as e:
    print(e)
    print("You have to start this file with the command:\
          cle-virtual-coach learn_weights.py")
    raise e

sim = vc.launch_experiment('ExDHBPPrak_Motion')
last_restart = time.time()
try:
    # If you can't keep your logging diarrhea in check, I will not respect your private variables
    sim._Simulation__logger.setLevel(logging.WARNING)
    # Additionally, you may want to change line 281 of nrp_virtual_coach/simulation.py to
    # if 'state' in status and status['state'] in ['stopped', 'halted']:
    # to avoid additional (unproblematic) error messages
    genetic_search()
finally:
    logger.info("Stopping simulation.")
    sim.stop()
