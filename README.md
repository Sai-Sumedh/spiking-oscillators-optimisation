# spiking-oscillators-optimisation
A network of coupled spiking neurons are used as oscillators to solve optimisation problems.

The relative settling phase encodes solution to graphical optimisation problems like vertex colouring.

The connectivity metrics- synaptic weights and timescales - resulting in phase repulsion are obtained

The files:
1. functions_sim.py
2. neuron_layer.py
3. neuron_network.py

contain functions to implement Leaky Integrate and Fire neurons and couple them using synaptic connections.

The files:
1. experiments_spkosc2.ipynb
2. experiments_spkosc3.ipynb
3. experiments_spkosc4.ipynb
4. experiments_spkosc5.ipynb
5. experiments_spkosc6.ipynb

use the above functions to perform various experiments and observe the effect of coupling on the settling phase

The file experiments_spkosc6.ipynb solves vertex colouring using a network of symmetrically coupled spiking neuron oscillators on simple graphs
