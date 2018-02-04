# -*- coding: utf-8 -*-
"""
This is a minimal brain with 2 neurons connected together.
"""
# pragma: no cover

__author__ = 'Felix Schneider'

from hbp_nrp_cle.brainsim import simulator as sim
import numpy as np

sensors = sim.Population(3, cellclass=sim.IF_curr_exp())
actors = sim.Population(6, cellclass=sim.IF_curr_exp())

# best weights so far:
weights = np.array([ 1.2687,  0.9408,  4.0275,  0.4076,  4.9567,  0.6792,  4.9276,
        3.8688,  2.1914,  4.1219,  0.9874,  0.3526,  3.5533,  3.8544,
        0.0482,  1.7837,  0.5833,  4.221 ])
weights = weights.reshape(3, 6)
# weights = np.random.rand(3, 6) * 5

projection = sim.Projection(sensors, actors, sim.AllToAllConnector(), sim.StaticSynapse(weight=weights))

circuit = sensors + actors
