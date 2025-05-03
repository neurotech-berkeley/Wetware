import time
import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime
from mea_integration import IntegratedMEAInterface
from openai_integration import IntegratedOpenAIGymAPI

def save_episode_data(episode_steps, episode_rewards, timestamp):
    """Save episode data to a CSV file"""
    filename = f"cartpole_episode_data_{timestamp}.csv"
    
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Steps', 'Reward'])
        
        for i in range(len(episode_steps)):
            writer.writerow([i+1, episode_steps[i], episode_rewards[i]])
    
    print(f"Episode data saved to {filename}")

def plot_episode_data(episode_steps, episode_rewards, timestamp):
    """Plot episode steps and rewards"""
    plt.figure(figsize=(12, 5))
    
    # Plot steps per episode
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(episode_steps)+1), episode_steps, marker='o')
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.grid(True)
    
    # Plot rewards per episode
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(episode_rewards)+1), episode_rewards, marker='o', color='orange')
    plt.title('Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"cartpole_performance_{timestamp}.png")
    print(f"Performance plot saved to cartpole_performance_{timestamp}.png")


def run_integrated_dishbrain():
    """Run the integrated DishBrain experiment."""
    # Initialize the integrated MEA interface
    mea_interface = IntegratedMEAInterface()
    
    # Connect to the MEA device
    if not mea_interface.connect_to_device():
        print("Failed to connect to MEA device. Exiting.")
        return
    
    try:
        # Start recording from the MEA
        if not mea_interface.start_recording():
            print("Failed to start recording. Exiting.")
            return
        
        # Initialize the OpenAI Gym API
        gym_api = IntegratedOpenAIGymAPI(mea_interface)
        
        # Number of episodes to run
        episodes = 100
        
        # Run episodes
        for episode in range(episodes):
            print(f"Starting Episode {episode + 1}/{episodes}")
            
            # Reset the environment
            gym_api.initialize_training()
            
            done = False
            step_count = 0
            
            # Run until episode is done
            while not done:
                # Wait a bit to ensure we have fresh neural data
                time.sleep(0.05)
                
                # Run one frame of CartPole and get state variables and reward
                pole_angle, pole_angular_velocity, reward, terminated = gym_api.run_single_frame()
                
                # Use reward/punishment to stimulate neurons accordingly
                mea_interface.stimulate_neurons(pole_angle, pole_angular_velocity, reward)
                
                # Check termination condition
                done = terminated
                step_count += 1
                
                # Print progress every 10 steps
                if step_count % 10 == 0:
                    print(f"Episode {episode + 1}, Step {step_count}, Reward so far: {gym_api.total_reward}")
            
            print(f"Episode {episode + 1} completed with total reward: {gym_api.total_reward}")
            
            # Short pause between episodes
            time.sleep(1.0)
    
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
    
    except Exception as e:
        print(f"Error during experiment: {e}")
    
    finally:
        # Clean up
        mea_interface.disconnect()
        print("Experiment completed.")

if __name__ == "__main__":
    run_integrated_dishbrain()
