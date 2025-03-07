# q laerning is a reinforcement learning (tests) of the type model free RL(may3tamadsh aela model ykhali agent 
# yat3alam wa7do b tajriba w alkhata2 has two types QL and Policy optimization) based on actions  mo5tamifa w 3ashwa2iya
# algo nkhdmo bihom fi hada model free hia deep q neural network  

from edge_sim_py import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from DQN import DQNAgent # algorithm for q learning ykhdam biha 
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_input_file(file_path):
    """Validate input JSON file structure."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return isinstance(data, dict)
    except (json.JSONDecodeError, FileNotFoundError, ValueError) as e:
        logging.error(f"Invalid input file: {e}")
        return False


def custom_collect_method(self) -> dict: # defined for EdgeServer objects
    """Enhanced power metrics collection."""
    return {   #This method collects metrics like
        "Object": f"EdgeServer_{self.id}",
        "Time Step": self.model.schedule.steps,
        "Instance ID": self.id,  # server id 
        "Power Consumption": self.dynamic_power()  #The power consumption is calculated using a dynamic_power method (cpu mem dusage)
    }

def dynamic_power(self):
    """Realistic power calculation with server variations"""
    # Hardware efficiency factors
    cpu_eff = 0.8 + (self.id % 5) * 0.05  # 0.8-1.0
    mem_eff = 0.7 + (self.id % 3) * 0.1   # 0.7-1.0
    disk_eff = 0.6 + (self.id % 2) * 0.2   # 0.6-0.8
    
    base_power = (
        np.log(self.cpu + 1) * 2.1 * cpu_eff +
        np.sqrt(self.memory) * 0.65 * mem_eff +
        (self.disk ** 0.33) * 0.35 * disk_eff
    )
    
    # Time-varying utilization
    utilization = 0.6 + 0.3 * np.sin(self.model.schedule.steps/50)
    # includes some time-varying utilization and random noise to make it realistic.
    return round(base_power * utilization * 0.75 + np.random.uniform(0, 1.5), 2)

EdgeServer.dynamic_power = dynamic_power  # added to the EdgeServer class

# Global variables
global agent, power_history  # agent (a DQNAgent) and power_history (a list to track total power over time)
power_history = []

def my_algorithm(parameters):  # The main algorithm is my_algorithm, which handles service migration.
    """Improved migration logic with diversity"""
    current_step = parameters['current_step']
    # lists all edge servers
    edge_servers = [es for es in EdgeServer.all() if es is not None]
    if not edge_servers:
        return

    # Build state vectors for each server's CPU, memory, disk, and power.
    state_vectors = []
    for es in edge_servers:
        state_vectors.extend([
            es.cpu,
            es.memory,
            es.disk,
            es.dynamic_power()
        ])
    
    # Get actions for all services
    # for each service, it uses the DQN agent to choose an action (which server to migrate to)
    for service in Service.all():
        if not service.being_provisioned:
            try:
                state_array = np.array(state_vectors, dtype=np.float32)
                action = agent.choose_action(state_array)
                target_server = edge_servers[action]
                
                current_power = service.server.dynamic_power() if service.server else float('inf')
                target_power = target_server.dynamic_power()
                
                if target_power < current_power * 0.85:
                    service.provision(target_server=target_server)
                    logging.info(f"Migrated to {target_server} (Δ {current_power-target_power:.2f}W)")
            except Exception as e:
                logging.error(f"Service {service.id} error: {e}")

    # Update power tracking
    current_total = sum(es.dynamic_power() for es in edge_servers)
    power_history.append(current_total)
    
    # Decay exploration rate with manual minimum
    agent.epsilon = max(agent.epsilon * 0.995, 0.15)

def stopping_criterion(model: object):
    return model.schedule.steps == 1000   # stops the simulation after 1000 steps

if __name__ == "__main__":
    # Initialize components
    input_file = "sample_dataset3.json"
    if not validate_input_file(input_file):
        raise ValueError("Invalid input file")
    
    simulator = Simulator(
        tick_duration=1,
        tick_unit="seconds",
        stopping_criterion=stopping_criterion,
        resource_management_algorithm=my_algorithm,
    )
    simulator.initialize(input_file=input_file)
    
    EdgeServer.collect = custom_collect_method

    # Initialize DQN agent with correct parameters
    valid_servers = [es for es in EdgeServer.all() if es is not None]
    state_dim = len(valid_servers) * 4
    
    agent = DQNAgent(
        state_dim=state_dim,  # num_servers × 4 (4 metrics per server)
        action_dim=len(valid_servers), # Equal to number of servers
        hidden_dim=128,  # Hidden layer: 128 neurons
        lr=0.0005,
        gamma=0.97, # Discount factor khasm 
        epsilon_decay=0.999
    )

    # Run simulation
    try:
        simulator.run_model()
    except Exception as e:
        logging.error(f"Simulation failed: {e}")
        exit(1)

    logs = pd.DataFrame(simulator.agent_metrics["EdgeServer"])
    print(logs)

    df = logs
    edge_server_ids = df['Instance ID'].unique()
    
    # Set the style for scientific publication
    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.figsize'] = (10, 12)
    plt.rcParams['figure.dpi'] = 300

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[2, 1])

    # Plot 1: Individual Server Power Consumption
    colors = ['b', 'g', 'r', 'c', 'm', 'y']  # Different color for each server
    for idx, server_id in enumerate(edge_server_ids):
        server_data = df[df['Instance ID'] == server_id]
        
        # Calculate mean and standard deviation
        mean_power = server_data['Power Consumption'].mean()
        std_power = server_data['Power Consumption'].std()
        
        # Plot with basic matplotlib
        ax1.plot(server_data['Time Step'], 
                server_data['Power Consumption'],
                color=colors[idx % len(colors)],
                label=f'Server {server_id}\nμ={mean_power:.2f}W\nσ={std_power:.2f}W')

    ax1.set_title('Individual Server Power Consumption Over Time', fontsize=14, pad=20)
    ax1.set_xlabel('Time Steps', fontsize=12)
    ax1.set_ylabel('Power Consumption (W)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Plot 2: Total System Power
    total_mean = np.mean(power_history)
    total_std = np.std(power_history)

    ax2.plot(range(len(power_history)), 
            power_history,
            color='darkgreen',
            label=f'Total Power\nμ={total_mean:.2f}W\nσ={total_std:.2f}W')

    # Add trend line
    z = np.polyfit(range(len(power_history)), power_history, 1)
    p = np.poly1d(z)
    ax2.plot(range(len(power_history)), 
             p(range(len(power_history))), 
             'r--', 
             label=f'Trend (slope={z[0]:.2f}W/step)')

    ax2.set_title('Total System Power Consumption', fontsize=14, pad=20)
    ax2.set_xlabel('Time Steps', fontsize=12)
    ax2.set_ylabel('Total Power (W)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add statistical summary
    stats_text = (f"Simulation Summary:\n"
                 f"Duration: {len(power_history)} steps\n"
                 f"Peak Power: {max(power_history):.2f}W\n"
                 f"Min Power: {min(power_history):.2f}W\n"
                 f"Efficiency Gain: {((max(power_history)-min(power_history))/max(power_history)*100):.1f}%")

    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))

    # Adjust layout
    plt.tight_layout()

    # Save plots
    plt.savefig("Power_Analysis_Results.png", dpi=300, bbox_inches='tight')
    plt.savefig("Power_Analysis_Results.pdf", bbox_inches='tight')
    plt.show()

    # Print statistical analysis
    print("\nStatistical Analysis Summary:")
    print("============================")
    print(f"Total Simulation Steps: {len(power_history)}")
    print(f"System Peak Power: {max(power_history):.2f}W")
    print(f"System Minimum Power: {min(power_history):.2f}W")
    print(f"Average Power Consumption: {total_mean:.2f}W")
    print(f"Standard Deviation: {total_std:.2f}W")
    print(f"Power Reduction: {((max(power_history)-min(power_history))/max(power_history)*100):.1f}%")



# Execution Flow
#Validate input

#Initialize simulator with custom components

#Configure RL agent

#Run simulation with power-aware migrations

#Process metrics

#Generate analytical visualizations