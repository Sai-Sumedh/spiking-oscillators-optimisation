import numpy as np
import random
import time
import matplotlib.pyplot as plt
from neuron_layer import NeuronLayerX
from neuron_network import NeuronNetwork
import warnings


def run_simulation(w_cross, w_self, i_ext_mag, synaptic_waveform, num_steps, num_neurons=2, v_initial= np.zeros((2,1))):
    nl = NeuronLayerX('nl',num_neurons,synaptic_waveform)
    NN = NeuronNetwork()
    NN.add_layer(nl)
    NN.create_connections([(nl, nl)])
    nl.Fanout_weights[0] = np.array([[w_self, w_cross], [w_cross, w_self]]) # cause of error: Fanout_weights is a LIST.
    # index refers to the index of output connection, so assign array to the first element in the list
    N = num_steps
    iext = i_ext_mag*np.ones((2,N))
    # iext[1,0:iext_delay_n2] = 0   # delay the synaptic current to neuron N2

    NN.reset_v_i_x()
    # set initial conditions
    nl.voltages = np.copy(v_initial) # removed for now
    v = np.zeros((N,2))
    s = np.zeros((N,2))
    c = np.zeros((N,2))
    i_in = np.zeros((N,2))

    for ti in range(N):
        v[ti, :] = nl.voltages.reshape((-1,))
        s[ti, :] = nl.activations.reshape((-1,))
        c[ti, :] = nl.i_presyn_current.reshape((-1,))
        i_in[ti, :] = nl.input_currents.reshape((-1,))
        # print(Z.i_presyn_current)
        NN.compute_v()
        NN.compute_i()
        # print(iext[:,tz].shape)
        nl.input_currents = nl.input_currents + iext[:,ti].reshape(nl.input_currents.shape)
        # print(nl.input_currents)
        # the crucial step: adding external currents after compute_i (since compute_i resets current initially)

    # plt.plot(s[:, 0])
    # plt.plot(s[:, 1])
    # plt.xlabel("time (ms)")
    # plt.ylabel("value")
    # plt.title("Spikes of neurons ")
    output = {'voltage': v, 'spike': s, 'presynaptic_current': c, 'input_current': i_in, 'external_current': iext,
              'neuron_layer': nl, 'neuron_network': NN}
    return output


def settling_phase(spike1, spike2):
    spike_times1 = np.argwhere(spike1)
    spike_times2 = np.argwhere(spike2)
    if spike_times1.size>0 and spike_times2.size>0:
        T1 = spike_times1[-1]-spike_times1[-2]
        T2 = spike_times2[-1]-spike_times2[-2]
        if np.abs(T1-T2)>2:
            warnings.warn("Time period of oscillation of 1,2 differs by more than 2ms")
        T = T1
        delT_12 = np.abs(spike_times1[-1]-spike_times2[-1])
        delphi_raw = (delT_12/T)*360 # in degrees
        delphi = min(delphi_raw, 360-delphi_raw)
    else:
        warnings.warn("Warning: Some neurons don't spike: phase difference used =-100")
        delphi = 0
    return delphi


def v_init_sweep(v_values, w_cr, w_se, iext, syn_wav):
    n_neurons=2
    n_steps = 500
    n_v = len(v_values)
    # v_values = np.linspace(0, 0.5, n_v)
    phi_all = np.zeros((n_v, n_v))
    for m in range(n_v):
        for k in range(n_v):
            v_init = np.array([v_values[m], v_values[k]]).reshape((2,1))
            outputs = run_simulation(w_cross=w_cr, w_self=w_se, i_ext_mag=iext,synaptic_waveform=syn_wav, num_steps=n_steps, num_neurons=n_neurons, v_initial=v_init)
            v,s = outputs['voltage'], outputs['spike']
            phi = settling_phase(s[:,0], s[:,1])
            phi_all[m,k] = phi
    return phi_all


def plot_phase(phi_mat, axis_limits, x_name, y_name, plot_title):
    plt.imshow(phi_mat, extent=axis_limits, origin='lower', cmap='cividis', vmin=0, vmax=180)
    plt.colorbar()
    plt.title(plot_title)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()


def get_phase_t(spikes_1, spikes_2):
    """
    spikes_1, spikes_2 are binary arrays of length=num_timesteps
    They contain 1 when the corresponding neuron issued a spike
    """
    assert spikes_1.shape == spikes_2.shape
    n_timesteps = len(spikes_1)
    phi1 = np.zeros(spikes_1.shape)
    phi2 = np.zeros(spikes_2.shape)

    for t in range(n_timesteps):
        t_spk_most_recent1 = [np.amax(np.argwhere(spikes_1[:t])) if len(np.argwhere(spikes_1[:t])) is not 0 else 0]
        # print(f"most recent spike time={t_spk_most_recent1}")
        t_spk_just_after1 = [np.amin(np.argwhere(spikes_1[t:])+t) if len(np.argwhere(spikes_1[t:])) is not 0 else t_spk_most_recent1[0]+T1]
        # after the last spike, the period of oscillation is taken to equal previous period T
        # print(f"spike time just after={t_spk_just_after1}")
        T1 = t_spk_just_after1[0]-t_spk_most_recent1[0]
        phi1[t] = (360*(t-t_spk_most_recent1[0])/T1)%360
        # print(f"Time period:{T}")
        # print(f"Phase:{phi1[t]}")

        t_spk_most_recent2 = [np.amax(np.argwhere(spikes_2[:t])) if len(np.argwhere(spikes_2[:t])) is not 0 else 0]
        # print(f"most recent spike time={t_spk_most_recent2}")
        t_spk_just_after2 = [np.amin(np.argwhere(spikes_2[t:])+t) if len(np.argwhere(spikes_2[t:])) is not 0 else t_spk_most_recent2[0]+T2]
        # after the last spike, the period of oscillation is taken to equal previous period T
        # print(f"spike time just after={t_spk_just_after2}")
        T2 = t_spk_just_after2[0]-t_spk_most_recent2[0]
        phi2[t] = (360*(t-t_spk_most_recent2[0])/T2)%360
        # print(f"Time period:{T2}")
        # print(f"Phase:{phi2[t]}")

    # plt.plot((phi1-phi2)%360)
    delphi = (phi1-phi2)%360

    return delphi,phi1, phi2


def get_settling_time(phi_t):
    """
    gives output in ms (gives number of timesteps, each timestep is = 1ms )
    """
    phi_last = phi_t[-5:]
    phi_ss = np.sum(phi_last)/len(phi_last)

    ss_times = np.argwhere(np.abs(phi_t-phi_ss)<0.05*phi_ss) # within 5% of steady state value
    t_s = np.amin(ss_times)
    return t_s


import matplotlib.animation as animation
from matplotlib import rc

rc('animation', html='jshtml')


def phase_space_anim(v1, v2):
    plt.style.use('dark_background')

    fig = plt.figure()
    ax = plt.axes(xlim=(0, 1), ylim=(0, 1))
    line, = ax.plot([], [], lw=2)

    # initialization function
    def init():
        # creating an empty plot/frame
        line.set_data([], [])
        return line,

        # lists to store x and y axis points

    xdata, ydata = [], []

    # animation function
    def animate(i):
        # appending new points to x, y axes points list
        xdata.append(v1[i])
        ydata.append(v2[i])
        line.set_data(xdata, ydata)
        return line,

        # setting a title for the plot

    plt.title('Phase space')
    plt.xlabel('v_1')
    plt.ylabel('v_2')
    # hiding the axis details
    plt.axis('on')

    # call the animator
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=500, interval=20, blit=True)
    # plt.show()
    return anim


def run_simulation_noisy(w_cross, w_self, i_ext_mag, synaptic_waveform, num_steps,noise_var=0.2, num_neurons=2, v_initial=np.zeros((2,1))):
    nl = NeuronLayerX('nl',num_neurons,synaptic_waveform)
    NN = NeuronNetwork()
    NN.add_layer(nl)
    NN.create_connections([(nl, nl)])
    nl.Fanout_weights[0] = np.array([[w_self, w_cross], [w_cross, w_self]]) # cause of error: Fanout_weights is a LIST.
    # index refers to the index of output connection, so assign array to the first element in the list
    N = num_steps
    iext = i_ext_mag*np.ones((2,N)) + np.random.normal(0, noise_var, size=(2,N))
    # iext[1,0:iext_delay_n2] = 0   # delay the synaptic current to neuron N2

    NN.reset_v_i_x()
    # set initial conditions
    nl.voltages = np.copy(v_initial) # removed for now
    v = np.zeros((N,2))
    s = np.zeros((N,2))
    c = np.zeros((N,2))
    i_in = np.zeros((N,2))

    for ti in range(N):
        v[ti, :] = nl.voltages.reshape((-1,))
        s[ti, :] = nl.activations.reshape((-1,))
        c[ti, :] = nl.i_presyn_current.reshape((-1,))
        i_in[ti, :] = nl.input_currents.reshape((-1,))
        # print(Z.i_presyn_current)
        NN.compute_v()
        NN.compute_i()
        # print(iext[:,tz].shape)
        nl.input_currents = nl.input_currents + iext[:,ti].reshape(nl.input_currents.shape)
        # print(nl.input_currents)
        # the crucial step: adding external currents after compute_i (since compute_i resets current initially)

    # plt.plot(s[:, 0])
    # plt.plot(s[:, 1])
    # plt.xlabel("time (ms)")
    # plt.ylabel("value")
    # plt.title("Spikes of neurons ")
    output = {'voltage': v, 'spike': s, 'presynaptic_current': c, 'input_current': i_in, 'external_current': iext,
              'neuron_layer': nl, 'neuron_network': NN}
    return output


def v_init_sweep_noisy(v_values, w_cr, w_se, iext, syn_wav, noise_var, n_steps):
    n_neurons=2
    n_v = len(v_values)
    # v_values = np.linspace(0, 0.5, n_v)
    phi_all = np.zeros((n_v, n_v))
    for m in range(n_v):
        for k in range(n_v):
            v_init = np.array([v_values[m], v_values[k]]).reshape((2,1))
            outputs = run_simulation_noisy(w_cross=w_cr, w_self=w_se, i_ext_mag=iext,synaptic_waveform=syn_wav,noise_var=noise_var, num_steps=n_steps, num_neurons=n_neurons, v_initial=v_init)
            v,s = outputs['voltage'], outputs['spike']
            phi = settling_phase(s[:,0], s[:,1])
            phi_all[m,k] = phi
    return phi_all


def w_sweep_noisy(w_values, v_init, i_ext_mag,noise_var, synaptic_waveform):
    n_w = len(w_values) # 10 different weight values
    n_neurons=2
    # w_values = -np.linspace(0, 5, n_w)
    phi_all = np.zeros((n_w, n_w))
    n_steps = 500
    # v_0 = np.array([0.1, 0.2]).reshape((2, 1))
    for a in range(n_w):
        for b in range(n_w):
          if w_values[a]>-1 or w_values[b]<=w_values[a]/2:
            w_c = w_values[a]
            w_s = w_values[b]
            # print(f"w_cross={w_c}, w_self={w_s}, v_init={v_0}")
            outputs = run_simulation_noisy(w_cross=w_c, w_self=w_s, i_ext_mag=i_ext_mag, synaptic_waveform=synaptic_waveform,noise_var=noise_var, num_steps=n_steps, num_neurons=n_neurons,v_initial=v_init)
            v,s  = outputs['voltage'], outputs['spike']
            phi = settling_phase(s[:,0], s[:,1])
            phi_all[b,a] = phi
    return phi_all


def get_synaptic_waveform(tau1, tau2):
    dt = 1e-3
    duration = 10*max(tau1, tau2) # 10 times the longest time constant
    t = np.arange(0, duration, dt)
    h_syn = np.exp(-t/tau1)-np.exp(-t/tau2)
    h_syn = h_syn/np.amax(h_syn)
    plt.plot(t, h_syn)
    plt.xlabel("time")
    plt.ylabel("i")
    plt.title("Synaptic waveform h(t)")
    plt.show()
    return h_syn
