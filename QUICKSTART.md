# Wetware Brian2 Simulation - Quick Start Guide

## ✅ Setup Complete!

Your Brian2 simulation environment is ready to use.

## Running the Simulation

### Option 1: Quick Run (5 episodes, no visualization)
```bash
cd ~/Wetware
source venv/bin/activate
python3 -c "from run_cartpole_brian2_fixed import run_simulation; run_simulation(episodes=5, visualize=False)"
```

### Option 2: Full Run with Visualization (saves plot)
```bash
cd ~/Wetware
source venv/bin/activate
python3 run_cartpole_brian2_fixed.py
```
This runs 10 episodes and saves results to `brian2_cartpole_results.png`

### Option 3: Custom Configuration
```bash
cd ~/Wetware
source venv/bin/activate
python3 -c "
from run_cartpole_brian2_fixed import run_simulation
run_simulation(episodes=50, visualize=True)  # Run 50 episodes
"
```

## What's Running?

The simulation connects:
- **60 Spiking Neurons** (Brian2 AdEx model) split into left/right action groups
- **CartPole-v1** environment from Gymnasium
- **Sensory encoding**: Pole angle/velocity → Stimulation frequency
- **Motor decoding**: Spike counts → Left (0) or Right (1) actions
- **Reward/Punishment**: Coherent patterns for success, chaos for failure

## Current Performance

**Baseline**: ~9-10 steps (random behavior)
- This indicates the system is running but neurons need better tuning to learn

## Files

- `run_cartpole_brian2_fixed.py` - Main simulation (working version)
- `SocketPrototypes/spike.py` - Spike detection algorithms
- `SocketPrototypes/square.py` - Stimulation waveform generation
- `venv/` - Python virtual environment with all dependencies

## Dependencies Installed

- ✅ Brian2 2.9.0 (spiking neural network simulator)
- ✅ Gymnasium 1.2.2 (RL environments)
- ✅ NumPy 2.3.4
- ✅ Matplotlib 3.10.7
- ✅ SciPy 1.16.3

## Next Steps to Improve Performance

1. **Increase neuron stimulation strength** - Edit line 90 in `run_cartpole_brian2_fixed.py`
2. **Add STDP learning** - Implement synaptic plasticity between neurons
3. **Tune spike detection** - Adjust threshold in `spike.py:63`
4. **Add baseline comparison** - Implement DQN agent for comparison
5. **Visualize neural activity** - Plot spike rasters during episodes

## Troubleshooting

**If simulation crashes:**
```bash
cd ~/Wetware
source venv/bin/activate
pip install --upgrade brian2 gymnasium
```

**If matplotlib blocks:**
Use `visualize=False` parameter

**If neurons don't spike:**
- Increase stimulation current on line 90
- Check neuron parameters (lines 219-226)

## Original vs Fixed Files

- Original (broken): `run_cartpole_brian2.py`
- Fixed (working): `run_cartpole_brian2_fixed.py`

---

**Questions?** Check the Wetware analysis document or modify the simulation parameters!
