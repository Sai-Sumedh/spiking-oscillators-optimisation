import numpy as np
import random
import time
import matplotlib.pyplot as plt


class NeuronNetwork:

    def __init__(self, time_step=1e-3, gamma=0.3):

        self.layers = []
        self.FO_Connections = {}  # fanout connections of each layer
        self.dt = time_step
        self.FI_Connections = {}  # fan-in connections of each layer
        self.time = 0e-3
        self.input_layer = None
        self.output_layer = None
        self.average_readout = None
        self.gamma = gamma
        self.output_bias = None

    def add_layer(self, layer1, is_input_layer=False, is_output_layer=False):

        self.layers.append(layer1)
        layer1.dt = self.dt
        layer1.time = self.time
        self.FO_Connections[layer1] = layer1.Fanout_layers  # a list of fanout layers
        self.fan_in_all()

        if is_input_layer:
            self.input_layer = layer1
        if is_output_layer:
            self.output_layer = layer1
            self.output_bias = np.zeros(self.output_layer.voltages.shape)

    def obtain_fan_in(self, layer_x):

        fan_in = [k for k, v in self.FO_Connections.items() if layer_x in v]  # a list of layers which fanout to layer_x
        return fan_in

    def fan_in_all(self):

        for layer in self.layers:
            self.FI_Connections[layer] = self.obtain_fan_in(layer_x=layer)

    # def i_in(self, layer_z):
    # i_in computation is very tedious, can instead compute i_in everytime i_out is computed

    def create_connections(self, list_of_tuples):
        # creates a directed dense connection between layer1 (pre-synaptic) and layer2 (post-synaptic) for
        # (layer1,layer2) in list_of_tuples
        # layers 1 and 2 must already be added to the network
        for j in range(len(list_of_tuples)):
            layer1, layer2 = list_of_tuples[j]
            layer1.create_dense_connection(out_layer=layer2)
            if layer2 not in self.FO_Connections[layer1]:
                self.FO_Connections[layer1].append(layer2)
            if layer1 not in self.FI_Connections[layer2]:
                self.FI_Connections[layer2].append(layer1)

    def get_fanout_weights(self):
        w = [self.layers[z].Fanout_weights for z in range(len(self.layers))]
        return w

    def reset_v_i_x(self):
        for layer in self.layers:
            layer.reset_all_values()
        self.time = 0

    def reset_i_input(self):   # useful after computing v(t+1) in each timestep
        for layer in self.layers:
            layer.reset_i_in()

    def compute_v(self):
        for layer in self.layers:  # important to perform all voltage updates at once
            # to prevent mixing of values at t and t+1
            # all voltages are computed using values at t
            # input layer only issues spikes, so there is no need to compute voltages of it.
            if layer != self.input_layer:
                layer.compute_v_next()
            if layer == self.output_layer:
                layer.voltages = layer.voltages + self.output_bias

    def update_time(self):
        for layer in self.layers:
            layer.time += self.dt
        self.time += self.dt

    def compute_i(self):  # should be used only after compute_v, since I(t+1) is to be computed from v(t+1)
        self.reset_i_input()  # first, reset the input currents of all layers
        for layer in self.layers:
            layer.compute_i_next()
        # since compute_v is done first, compute_i later - time must be updated after compute_i
        self.update_time()

    # def softmax_output(self):
    #     z = np.sum(np.exp(self.average_readout))
    #     ans = np.exp(self.average_readout)/z
    #     return ans
    # not useful now

    def compute_loss(self, loss_function, y):
        # loss function= 'cross_entropy' or 'mse', y=expected values
        # layer_output's softmax output is assumed to be a numpy row array
        # this function computes the loss for one training example only
        predict = self.output_layer.voltages
        loss_computed = 0
        if loss_function == 'cross_entropy':
            loss_computed = -np.sum(y*np.log(predict) + (1-y)*np.log(1-predict))  # elementwise product
        elif loss_function == 'mse':
            loss_computed = (1/2)*np.sum(np.square(y-predict))
        return loss_computed
