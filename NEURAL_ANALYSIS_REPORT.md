# Wetware DishBrain - Comprehensive Neural Analysis Report

**Date:** November 11, 2025
**Simulation:** 50 Episodes of CartPole-v1 with Brian2 Spiking Neural Networks
**Neural Architecture:** 60 AdEx neurons (30 left, 30 right hemispheres)

---

## Executive Summary

The Brian2 simulation successfully ran 50 episodes with comprehensive neural activity tracking. The system demonstrates **functional connectivity** between spiking neurons and the CartPole environment, with **over 1.3 million spikes recorded** across both hemispheres. However, performance remains at **baseline random levels** (~9.4 steps average), indicating the absence of learning mechanisms.

---

## Performance Metrics

### Overall Statistics (50 Episodes)

| Metric | Value |
|--------|-------|
| **Episodes Run** | 50 |
| **Average Reward** | 9.38 ± 0.77 |
| **Average Steps** | 9.38 ± 0.77 |
| **Best Performance** | 11 steps |
| **Worst Performance** | 8 steps |
| **Median Reward** | 10.0 |

### Key Observations

1. **Stable Performance**: Low variance (σ = 0.77) indicates consistent behavior
2. **No Learning Curve**: Moving average remains flat across all 50 episodes
3. **Random Baseline**: Performance matches random action selection (~8-12 steps)
4. **No Improvement**: Episodes 1-10 perform identically to episodes 41-50

---

## Neural Activity Analysis

### Spike Statistics

| Population | Total Spikes | Avg Firing Rate |
|------------|--------------|-----------------|
| **Left Neurons** | 468,177 | ~15,606 spikes/episode |
| **Right Neurons** | 870,812 | ~29,027 spikes/episode |
| **Total** | 1,338,989 | ~44,633 spikes/episode |

### Critical Finding: Right Hemisphere Bias

The **right hemisphere fires 1.86× more** than the left (870k vs 468k spikes), which explains why:
- **88% of actions taken were "Right" (action = 1)**
- Only 12% were "Left" (action = 0)

This massive imbalance suggests:
- **Non-uniform stimulation**: Right neurons receive stronger or more frequent input
- **Asymmetric network initialization**: Possible bias in initial conditions
- **Decoding bias**: Spike counting threshold may favor right hemisphere

---

## Visualization Analysis

### 1. Performance Analysis (`neural_analysis_performance.png`)

**Top Panel - Learning Progress:**
- Flat reward trajectory across all episodes
- No upward trend in 10-episode moving average
- Performance plateaus immediately at random baseline

**Reward Distribution:**
- Narrow Gaussian centered at 9-10 steps
- Mean (red) = 9.38, Median (green) = 10.0
- Extremely tight clustering indicates no outlier "breakthrough" episodes

**Action Distribution:**
- **CRITICAL**: 88% right actions vs 12% left actions
- Severe imbalance explains lack of performance
- CartPole requires balanced left/right actions to succeed

**Firing Rate Distribution:**
- Both hemispheres show near-zero spike detection
- Boxplot indicates spike counting algorithm detects almost no activity
- **ISSUE**: Raw spikes exist (1.3M total) but aren't being decoded properly

**Pole Angle Trajectory:**
- Random oscillations with no stabilization
- Angles rapidly exceed ±0.2 rad threshold
- No evidence of corrective actions

**Stimulation Current:**
- Occasional large spikes (up to 1.3 units)
- Mostly low baseline (~0.2-0.3)
- Spikes correlate with punishment signals (high pole angles)

**Neural Activity Over Time:**
- Both hemispheres show near-zero spike counts in decoding
- Contradicts raw spike data showing 1.3M spikes
- **CRITICAL BUG**: Spike detection (MADs algorithm) is not capturing Brian2 spike events

---

### 2. Spike Raster Plot (`neural_analysis_raster.png`)

**Left Hemisphere (Blue):**
- Dense, uniform firing across all 30 neurons
- High-frequency bursting pattern (appears as solid bands)
- Continuous activity throughout entire simulation (~20 seconds)
- All neurons participate equally

**Right Hemisphere (Orange):**
- Even denser firing than left hemisphere
- More neurons fire at higher rates
- Consistent sustained activity
- Explains higher spike count (870k vs 468k)

**Episode Boundaries (Red Dashes):**
- No visible change in firing patterns across episodes
- Neurons don't "learn" or modulate activity between episodes
- Further evidence of absence of plasticity

**Interpretation:**
- Neurons are **too active** - firing almost continuously
- This creates a **signal-to-noise problem**: every stimulation triggers spikes
- No differential response to different pole angles
- The system has become a **binary oscillator** rather than a controller

---

### 3. Neural Activity Heatmaps (`neural_analysis_heatmaps.png`)

**Top Row - Activity Over Time:**
- Uniform orange coloring = constant low-level activity
- No temporal structure or patterns
- Both hemispheres show identical flat profiles
- **Missing**: Burst-pause cycles, activity modulation, state-dependent firing

**Bottom Row - Activity by Action:**
- **SHOCKING RESULT**: Both left AND right neurons show identical activity regardless of action taken
- Histograms overlap completely for "Action Left" vs "Action Right"
- All spike counts cluster at exactly 0.0
- **This confirms the decoding bug**: Spike counting returns zero despite 1.3M raw spikes

**What This Means:**
- The neural decoder (MADs + spike counting) is broken
- It returns zero for all frames, causing random action selection
- Actions are determined by noise, not neural activity
- The "right bias" comes from the tie-breaking rule: `action = 1 if left_count <= right_count`

---

## Root Cause Analysis

### Why Performance Is Random

1. **Spike Detection Failure**
   - The `MADs()` algorithm from `spike.py` is designed for continuous voltage traces
   - Brian2 outputs discrete spike events, not continuous data
   - The synthetic voltage buffer (100 amplitude spikes) doesn't match real MEA data statistics
   - High-pass filtering and MAD calculations return near-zero values

2. **Action Selection Breakdown**
   - With zero spike counts detected, the comparison `left_count > right_count` always fails
   - Default action becomes 1 (right) in most cases
   - This creates the 88% right bias
   - CartPole fails quickly without balanced corrections

3. **No Learning Mechanism**
   - Vanilla AdEx neurons have no synaptic plasticity
   - STDP was defined in original code but never implemented
   - Stimulation patterns don't encode meaningful state information
   - Reward signals don't modulate neuron parameters

4. **Stimulation-Response Mismatch**
   - High stimulation current (0.01 nA per neuron) causes constant firing
   - Neurons saturate - they can't increase firing when needed
   - Lack of dynamic range means all states look the same neurally

---

## Recommendations for Improvement

### High Priority - Fix Decoding

**Option A: Replace Spike Counting**
```python
# Instead of MADs + threshold, directly count Brian2 spikes
left_spike_count = len(self.left_spikes.i)
right_spike_count = len(self.right_spikes.i)
action = 0 if left_spike_count > right_spike_count else 1
```

**Option B: Use Firing Rate Windows**
```python
# Count spikes in recent time window
recent_left = sum(1 for t in left_spikes.t if t > (current_time - window))
recent_right = sum(1 for t in right_spikes.t if t > (current_time - window))
```

### High Priority - Reduce Stimulation

**Current:** 0.01 nA (causes saturation)
**Recommended:** 0.001-0.005 nA (allows dynamic range)

```python
stim_current = np.mean(np.abs(stim_wave)) * 0.002 * b2.nA  # 5x reduction
```

### Medium Priority - Add Learning

**Implement STDP:**
```python
# Add synapses between left and right populations
synapses = b2.Synapses(left_neurons, right_neurons,
    model='w : 1',
    on_pre='v_post += w*mV')
synapses.connect(p=0.1)

# Add reward-modulated STDP rule
dopamine_signal = reward - baseline
synapses.w += dopamine_signal * (spike_timing_difference)
```

**Implement Homeostatic Plasticity:**
```python
# Gradually adjust neuron excitability based on performance
if episode_reward > threshold:
    neuron.gL *= 0.99  # Make less excitable on success
else:
    neuron.gL *= 1.01  # Make more excitable on failure
```

### Medium Priority - Improve State Encoding

**Current Issues:**
- Angle and velocity encoded independently
- Frequency modulation range too narrow (10-30 Hz)
- No temporal structure

**Recommendations:**
```python
# Population coding: distribute state across multiple neurons
for i, neuron in enumerate(left_neurons):
    # Each neuron has preferred angle
    preferred_angle = (i / len(left_neurons)) * 0.4 - 0.2
    distance = abs(pole_angle - preferred_angle)
    stim_strength = exp(-distance**2 / (2 * sigma**2))
    neuron.I = base_current * stim_strength
```

### Low Priority - Add Visualization

**Real-time Monitoring:**
- Live spike raster during training
- Episode reward tracking
- Action distribution over time
- Synaptic weight evolution (if STDP added)

---

## Comparison to Random Baseline

| Metric | DishBrain | Random Agent | Trained DQN |
|--------|-----------|--------------|-------------|
| Avg Steps | 9.38 | ~9-11 | ~200-500 |
| Std Dev | 0.77 | ~1-2 | ~50-100 |
| Best | 11 | ~15 | 500 |

**Conclusion**: Current system performs identically to random action selection.

---

## Scientific Insights

### What We Learned

1. **Brian2 Integration Works**: Successfully interfaced spiking neurons with Gym environment
2. **Large-Scale Simulation**: Tracked 1.3M spikes across 50 episodes without crashes
3. **Visualization Pipeline**: Comprehensive tracking and plotting infrastructure operational
4. **System Bottleneck Identified**: Decoding is the weak link, not neural simulation

### Biological Plausibility

**Pros:**
- AdEx model captures realistic spike dynamics
- Stimulus-response loop mimics sensory-motor pathways
- Punishment via noise reflects aversive learning

**Cons:**
- No synaptic plasticity (real neurons learn)
- Continuous saturation firing (real neurons have refractory periods)
- Immediate stimulation (real neurons have propagation delays)
- Binary left/right split (real brains have distributed representations)

---

## Next Experiment Proposals

### Experiment 1: Fixed Decoding (1 day)
- Implement direct Brian2 spike counting
- Re-run 50 episodes
- **Hypothesis**: Performance will improve to 15-25 steps

### Experiment 2: Reduced Stimulation (1 day)
- Lower current by 5×
- Measure firing rate vs pole angle correlation
- **Hypothesis**: Dynamic range will emerge

### Experiment 3: STDP Learning (3-5 days)
- Add reward-modulated STDP
- Run 500 episodes
- **Hypothesis**: Gradual learning curve, reaching 50-100 steps by episode 500

### Experiment 4: Population Coding (2-3 days)
- Implement Gaussian tuning curves
- Compare to current frequency encoding
- **Hypothesis**: Better state representation → better performance

---

## Files Generated

1. `run_cartpole_brian2_enhanced.py` - Enhanced simulation with tracking
2. `neural_analysis_performance.png` - Performance metrics (301 KB)
3. `neural_analysis_raster.png` - Spike raster plots (141 KB)
4. `neural_analysis_heatmaps.png` - Activity heatmaps (96 KB)
5. `NEURAL_ANALYSIS_REPORT.md` - This report

---

## Conclusion

The Wetware DishBrain simulation successfully demonstrates a **functional neuromorphic computing pipeline** connecting spiking neural networks to reinforcement learning environments. However, the system currently operates at **random baseline performance** due to a **critical decoding bug** that prevents neural activity from influencing actions.

**The good news**: The infrastructure works! With proper spike decoding and reduced stimulation, the system should achieve meaningful performance. Adding STDP would enable true learning.

**The opportunity**: This is one of the few open-source implementations of a closed-loop "organoid-in-the-loop" system using Brian2. Fixing the decoding issue would make this a valuable educational and research tool.

**Recommended Next Steps:**
1. Fix spike decoding (30 min fix)
2. Test with reduced stimulation (1 hour)
3. If successful, implement STDP (1-2 days)
4. Write paper: "From Random to Learning: Engineering Neuroplasticity in Simulated DishBrain Systems"

---

**Report Generated:** November 11, 2025
**Total Simulation Time:** ~40 seconds (50 episodes)
**Total Spikes Recorded:** 1,338,989
**Analysis Complete** ✓
