from brian2 import prefs
prefs.codegen.target = "numpy"
import numpy as np
import gymnasium as gym
import brian2 as b2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from brian2_virtual_mea import VirtualMEA
from OpenAIGymAPI import OpenAIGymAPI
from square import generate_stim_wave

# Start Brian2 simulation scope
b2.start_scope()

# Set up simulation parameters
num_channels = 60
buffer_size = 100
simulation_dt = 0.1 * b2.ms

# ⚠️ Define all constants BEFORE neuron_eqs and NeuronGroup
EL = -70 * b2.mV    #Resting Potential
VT = -50 * b2.mV    #Spike threshold
Delta_T = 2 * b2.mV #Slope factor for spike initiation
gL = 10 * b2.nsiemens #Leak conductance
C = 200 * b2.pfarad #Membrane capacitance
a = 2 * b2.nsiemens #Adaptation conductance
tau_w = 100 * b2.ms #Adaptation time constant
b = 0.05 * b2.nA #Adaptation increment after spike

namespace = {
    'EL': EL,
    'VT': VT,
    'Delta_T': Delta_T,
    'gL': gL,
    'C': C,
    'a': a,
    'tau_w': tau_w,
    'b': b
}

# Define neuron equations
neuron_eqs = '''
dv/dt = (gL*(EL-v) + gL*Delta_T*exp((v-VT)/Delta_T) - w + I)/C : volt
dw/dt = (a*(v-EL) - w)/tau_w : amp
I : amp
'''

# Create neuron groups
left_neurons = b2.NeuronGroup(num_channels // 2, neuron_eqs,
                              threshold='v > -50*mV',
                              reset='v = -70*mV; w += b',
                              method='euler',
                              namespace=namespace)

right_neurons = b2.NeuronGroup(num_channels // 2, neuron_eqs,
                               threshold='v > -50*mV',
                               reset='v = -70*mV; w += b',
                               method='euler',
                               namespace=namespace)

# Initialize neuron states
#v = membrane potential
#w = adaptation current (how neurons become less excitable immediately after firing)
for neurons in [left_neurons, right_neurons]:
    neurons.v = EL
    neurons.w = 0 * b2.pA

# Set up spike monitors
# Each time the membrane potential crosses the defined threshold, the spike monitor logs:
    # The index of the neuron that spiked
    # The simulation time at which the spike occured
left_spikes = b2.SpikeMonitor(left_neurons)
right_spikes = b2.SpikeMonitor(right_neurons)

# Build network
network = b2.Network(left_neurons, right_neurons, left_spikes, right_spikes)

# Virtual MEA
virtual_mea = VirtualMEA(left_neurons, right_neurons, left_spikes, right_spikes, network)

# Initialize the OpenAI Gym environment
env = gym.make('CartPole-v1')
openai_gym_api = OpenAIGymAPI(virtual_mea, num_channels, buffer_size)

# Run episodes
episodes = 100
for episode in range(episodes):
    print(f"Starting Episode {episode + 1}/{episodes}")
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    # After each step:
    # System reads the pole angle and velocity
    # Values are transformed into input currents
    # Neurons respond by firing spikes
    # Spiking patterns determine the action, and the CartPole environment updates accordingly 
    
    while not done:
        pole_angle, pole_angular_velocity, reward, terminated = openai_gym_api.run_single_frame(None)
        virtual_mea.stimulate_neurons(pole_angle, pole_angular_velocity, reward, None)
        done = terminated
        total_reward += reward

    print(f"Episode {episode + 1} completed with total reward: {total_reward}")

# Visualization showing animation of cartpole and neural activity
def visualize_dishbrain_simulation(env, virtual_mea):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Initialize cartpole environment
    state, _ = env.reset()

    # Initialize neural activity plot
    neural_data = np.zeros((num_channels, buffer_size))
    im = ax2.imshow(neural_data, aspect='auto', cmap='hot', vmin=0, vmax=100)
    ax2.set_title('Neural Activity')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Channel')

    # Initialize cartpole plot
    cart_width = 0.2
    cart_height = 0.1
    pole_length = 1.0
    cart = plt.Rectangle((state[0] - cart_width/2, -cart_height/2), cart_width, cart_height, color='blue')
    pole = plt.Line2D([state[0], state[0] + pole_length * np.sin(state[2])],
                      [0, pole_length * np.cos(state[2])], lw=3, color='red')
    ax1.add_patch(cart)
    ax1.add_line(pole)
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_title('CartPole Environment')

    def update(frame):
        # Run one step of simulation
        pole_angle, pole_angular_velocity, reward, terminated = openai_gym_api.run_single_frame(None)
        
        # Update neural data visualization
        neural_data = virtual_mea.read_neural_data_buffer(num_channels, buffer_size)
        im.set_array(neural_data)

        # Update cartpole visualization
        cart.set_x(state[0] - cart_width/2)
        pole.set_xdata([state[0], state[0] + pole_length * np.sin(state[2])])
        pole.set_ydata([0, pole_length * np.cos(state[2])])

        return [im, cart, pole]

    ani = FuncAnimation(fig, update, frames=500, interval=20, blit=True)
    plt.tight_layout()
    plt.show()
    
    # Save animation for presentation
    ani.save('dishbrain_simulation.mp4', writer='ffmpeg', fps=30)

# Call visualization
visualize_dishbrain_simulation(env, virtual_mea)

def plot_learning_progress(rewards, spike_rates, episodes):
    """Plot learning progress over episodes"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot rewards
    ax1.plot(range(1, episodes+1), rewards)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Learning Progress: Reward per Episode')
    
    # Plot spike rates
    ax2.plot(range(1, episodes+1), spike_rates)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Spike Rate (Hz)')
    ax2.set_title('Neural Activity During Learning')
    
    plt.tight_layout()
    plt.savefig('learning_progress.png')
    plt.show()
    
    
# Define STDP synapses between neurons
stdp_eqs = '''
w : 1
dapre/dt = -apre/taupre : 1 (event-driven)
dapost/dt = -apost/taupost : 1 (event-driven)
'''

stdp_pre = '''
apre += Apre
w = clip(w + apost, 0, wmax)
'''

stdp_post = '''
apost += Apost
w = clip(w + apre, 0, wmax)
'''

# Create synapses between neurons with STDP
synapses = b2.Synapses(left_neurons, right_neurons, 
                      model=stdp_eqs,
                      on_pre=stdp_pre,
                      on_post=stdp_post)

# Connect neurons with initial random weights
synapses.connect(p=0.1)  # 10% connection probability
synapses.w = 'rand() * 0.5'