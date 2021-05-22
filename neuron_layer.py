import numpy as np
import random
import time
import matplotlib.pyplot as plt


class NeuronLayer:
    """
    This class implements a layer of neurons
    """
    def __init__(self, title, num_neurons,  num_excitatory=0, time_step=1e-3, refractory_period=5e-3,
                threshold_baseline=0.61, threshold_variation_scale=0,
                 v_decay_time_constant=20e-3, synaptic_delay=1e-3, threshold_decay_time_constant=700e-3):

        # all neurons in the layer are of the same model : lif or adaptive lif
        self.title = title  # name of the layer
        self.occupancy = num_neurons  # stores the total number of neurons in the layer
        self.n_excitatory = num_excitatory
        self.voltages = np.zeros(shape=(self.occupancy,1))
        self.activations = np.zeros(shape=(self.occupancy,1))  # stores the delta spikes (0 or 1/dt) of each neuron
        self.Fanout_layers = []  # the layers which this layer connects to
        self.Fanout_weights = []  # instead of dictionary, this will be a list of 2d arrays
        self.input_currents = np.zeros(shape=(self.occupancy, 1))  # 1 D array of input currents
        # list of spike traces of each neuron
        self.spike_times = [[-1] for i in range(self.occupancy)]  # list for each neuron in the layer
        # the first element of spike times of each neuron is -1
        self.vth = threshold_baseline * np.ones(shape=self.voltages.shape)  # initialize threshold to baseline
        # time constants, other constants (except vth, which can change in adaptive layer)
        self.time = 0e-3  # keeps track of time

        self.dt = time_step  # duration of one step
        # constants to work with
        self.vth_baseline = threshold_baseline
        self.beta = threshold_variation_scale  # if this term is nonzero, then neuron is adaptive lif
        self.tref = refractory_period
        self.tau_m = v_decay_time_constant
        if isinstance(synaptic_delay,float):
            self.d_syn = synaptic_delay
        # elif isinstance(synaptic_delay,list):
        #     assert len(synaptic_delay) == self.occupancy
        #     self.d_syn = np.array(synaptic_delay).reshape((-1,1))
        else:
            raise ValueError
        # can be a list, if different d_syn is desired for each neuron
        self.tau_a = threshold_decay_time_constant  # for adaptive neuron layer,
        # one which has threshold_variation_scale non-zero
        # tau_a can be given as a numpy array, if different adaptation time constants are desired,
        # with size num_neurons,1

    def create_dense_connection(self, out_layer):

        self.Fanout_layers.append(out_layer)  # since this layer is newly added, it will be the last element in the list
        # so, its index will be length of fanout layers - 1

        # initialize the fanout weights
        n_out = out_layer.occupancy
        n_in = self.occupancy
        if self.n_excitatory is None:  # unconstrained weights
            w0 = 6  # removing dt, since activation is now binary
            weights = w0 * (1/np.sqrt(n_in))*np.random.normal(0.0,1.0,size=(n_out,n_in))

        else:
            w_exc = np.abs(np.random.normal(loc=0.0, scale=1.0, size=(n_out, self.n_excitatory)))
            # excitatory fanout weights
            w_inh = -1 * np.abs(np.random.normal(loc=0.0, scale=1.0, size=(n_out, n_in - self.n_excitatory)))
            w0 = 6  # w0 in paper, ignore r_mem since it gets cancelled in voltage computation
            weights = w0 * (1 / np.sqrt(n_in)) * np.concatenate((w_exc, w_inh), axis=1)  # concatenate along columns
        self.Fanout_weights.append(weights)
        # no need to initialize output currents, they can be computed anytime using fanout weights and activations

    def reset_all_values(self):
        self.voltages = np.zeros(shape=(self.occupancy, 1))
        self.activations = np.zeros(shape=(self.occupancy, 1))
        self.input_currents = np.zeros(shape=(self.occupancy, 1))
        self.spike_times = [[-1] for i in range(self.occupancy)]
        self.vth = self.vth_baseline * np.ones(shape=self.voltages.shape)
        self.time = 0.0

    def compute_i_next(self):
        activations_before_d_syn = np.zeros(shape=self.activations.shape)
        spiked_before_dsyn = [i for i,x in enumerate(self.spike_times) if self.time-(self.d_syn-self.dt) in x]
        activations_before_d_syn[spiked_before_dsyn] = 1  # activations considering synaptic delay
        out_current = [np.zeros(shape=self.Fanout_weights[i].shape) for i in range(len(self.Fanout_layers))]
        for i in range(len(self.Fanout_layers)):
            out_current[i] = np.dot(self.Fanout_weights[i],activations_before_d_syn.reshape((-1,1))).reshape(
                self.Fanout_layers[i].input_currents.shape)
            self.Fanout_layers[i].input_currents = self.Fanout_layers[i].input_currents + out_current[i]

    def reset_i_in(self):
        self.input_currents = np.zeros(shape=(self.occupancy, 1))

    def compute_v_next(self):
        # assumptions:
        # I_in(t), v(t), vth(t) (in case of adaptive lif) are known
        # uses these values at t to calculate v(t+1)
        # idea is to first update voltages of all layers WITHOUT updating output, input currents:
        # use values at t to compute voltages at t+1, THEN compute output currents of all layers
        # the below is a naive, loop over all neurons interpretation
        # update voltages of only those which are not in the refractory period
        recent_spike_times = np.array([self.spike_times[q][-1] for q in range(self.occupancy)]).reshape((-1,1))
        # most recent spike time
        # if no spike so far for a neuron, -1 is obtained
        alpha = np.exp(-self.dt / self.tau_m)
        not_in_refrac = self.time-recent_spike_times >= self.tref
        self.voltages[not_in_refrac] = \
            alpha*self.voltages[not_in_refrac] + (1-alpha)*self.input_currents[not_in_refrac]  # reset not needed,
        # since these are not in refractory period
        reset_voltage = (self.time-recent_spike_times < 2*self.dt) & (self.time-recent_spike_times > 0)
        # broadcasting takes care, neurons which have just spiked
        # in the previous timestep
        self.voltages[reset_voltage] = alpha*self.voltages[reset_voltage] - self.vth[reset_voltage]*self.activations[reset_voltage]
        # assuming self.activations have been correctly set to 1/dt, since the neuron has spiked
        # threshold update
        rho = np.exp(-self.dt / self.tau_a)
        self.vth = rho * self.vth + (1 - rho) * self.vth_baseline + (1 - rho) * self.beta * self.activations
        # spike issue
        just_spiked=(self.voltages>self.vth) & not_in_refrac
        self.activations[just_spiked] = 1
        need_activ_reset = np.logical_not(just_spiked)
        self.activations[need_activ_reset]=0
        just_spiked_indices = [i for i, x in enumerate(just_spiked) if x]
        for index in just_spiked_indices:
            self.spike_times[index].append(self.time)  # this for loop seems unavoidable

    def get_spike_times(self):
        spk_times = [self.spike_times[k][1:] for k in range(self.occupancy)]  # removes the initial -1
        return spk_times

    # def obtain_softmax(self):
    #     # computes softmax of voltages
    #     # to be used for output layer
    #     # returns a column of probabilities
    #     assert len(self.voltages) != 1
    #     z = np.sum(np.exp(self.voltages))
    #     softmax_values = np.exp(self.voltages)/z
    #     return softmax_values


class NeuronLayerX(NeuronLayer):    # adds synaptic waveform to neuron model
    def __init__(self, title, num_neurons, synaptic_waveform, num_excitatory=0, time_step=1e-3, refractory_period=5e-3,
                 threshold_baseline=1, threshold_variation_scale=0,
                 v_decay_time_constant=20e-3, synaptic_delay=1e-3, threshold_decay_time_constant=700e-3):
        super().__init__(title, num_neurons, num_excitatory=0, time_step=1e-3, refractory_period=5e-3,
                         threshold_baseline=1, threshold_variation_scale=0,
                         v_decay_time_constant=20e-3, synaptic_delay=1e-3, threshold_decay_time_constant=700e-3)

        self.synaptic_waveform = synaptic_waveform
        self.synaptic_waveform_length = len(synaptic_waveform)
        self.i_presyn_current = np.zeros((self.occupancy, 1))

    def compute_v_next(self):
        recent_spike_times = np.array([self.spike_times[q][-1] for q in range(self.occupancy)]).reshape((-1, 1))

        # most recent spike time
        # if no spike so far for a neuron, -1 is obtained
        alpha = np.exp(-self.dt / self.tau_m)
        not_in_refrac = self.time - recent_spike_times >= self.tref
        # print(not_in_refrac.shape)
        # print(self.input_currents.shape)
        self.voltages[not_in_refrac] = \
            alpha * self.voltages[not_in_refrac] + (1 - alpha) * self.input_currents[not_in_refrac]  # reset not needed,
        # since these are not in refractory period
        reset_voltage = (self.time - recent_spike_times < 2 * self.dt) & (self.time - recent_spike_times > 0)
        # broadcasting takes care, neurons which have just spiked
        # in the previous timestep
        v_reset = 0
        self.voltages[reset_voltage] = v_reset
        # assuming self.activations have been correctly set to 1/dt, since the neuron has spiked
        # threshold update
        rho = np.exp(-self.dt / self.tau_a)
        self.vth = rho * self.vth + (1 - rho) * self.vth_baseline + (1 - rho) * self.beta * self.activations
        # spike issue
        just_spiked = (self.voltages > self.vth) & not_in_refrac
        self.activations[just_spiked] = 1
        need_activ_reset = np.logical_not(just_spiked)
        self.activations[need_activ_reset] = 0
        just_spiked_indices = [i for i, x in enumerate(just_spiked) if x]
        for index in just_spiked_indices:
            self.spike_times[index].append(self.time)  # this for loop seems unavoidable

    def compute_i_next(self):
        # print("This is being executed")
        activations_before_d_syn = np.zeros(shape=self.activations.shape)

        spiked_before_dsyn = [i for i, x in enumerate(self.spike_times) if
                              len(x) > 1 and self.time - (self.d_syn) >= min(
                                  x[1:])]  # it has been more than synaptic delay
        # print(spiked_before_dsyn)
        i_pre_synaptic = np.zeros((self.occupancy, 1))
        # print(i_pre_synaptic)                                                             # time from first spike to self.time
        for neuroni in spiked_before_dsyn:
            spike_times_i = np.array(self.spike_times[neuroni])
            spike_times_i = spike_times_i[spike_times_i != -1]
            # print(spike_times_i)
            spike_times_d_syn_done = spike_times_i[
                spike_times_i <= (self.time - (self.d_syn))]  # spike times before synaptic delay
            # print(spike_times_d_syn_done)
            waveform_position_index = np.ceil(
                (self.time - (self.d_syn) - spike_times_d_syn_done) / self.dt)  # a vector having indices
            # print(waveform_position_index)
            spikes_within_synaptic_window = spike_times_d_syn_done[
                waveform_position_index < self.synaptic_waveform_length]
            # print(spikes_within_synaptic_window)
            valid_waveform_indices = waveform_position_index[waveform_position_index < self.synaptic_waveform_length]
            valid_waveform_indices = np.array(
                [int(x) - 1 for x in valid_waveform_indices])  # -1 since index must start from 0
            # print(valid_waveform_indices)
            if len(valid_waveform_indices) > 0:
                currents = self.synaptic_waveform[valid_waveform_indices]
                # print(currents)
                total_current_i = np.sum(currents)
                # print(total_current_i)
                i_pre_synaptic[neuroni,0] = total_current_i
                # print(i_pre_synaptic)

        self.i_presyn_current = i_pre_synaptic
        # if np.sum(self.i_presyn_current) != 0:
        #   print("Current is becoming nonzero")
        #   print(self.i_presyn_current)

        # print(L.i_presyn_current)

        activations_before_d_syn[spiked_before_dsyn] = 1  # activations considering synaptic delay
        postsyn_current = [np.zeros(shape=self.Fanout_layers[i].input_currents.shape) for i in
                           range(len(self.Fanout_layers))]
        for i in range(len(self.Fanout_layers)):
            postsyn_current[i] = np.dot(self.Fanout_weights[i], i_pre_synaptic)
            self.Fanout_layers[i].input_currents = self.Fanout_layers[i].input_currents + postsyn_current[i]

