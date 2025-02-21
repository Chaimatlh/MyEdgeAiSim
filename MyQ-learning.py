from edge_sim_py import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from DQN import DQNAgent
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

def custom_collect_method(self) -> dict:
    """Enhanced power metrics collection."""
    return {
        "Object": f"EdgeServer_{self.id}",
        "Time Step": self.model.schedule.steps,
        "Instance ID": self.id,
        "Power Consumption": self.dynamic_power()
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
    
    return round(base_power * utilization * 0.75 + np.random.uniform(0, 1.5), 2)

EdgeServer.dynamic_power = dynamic_power

# Global variables
global agent, power_history
power_history = []

def my_algorithm(parameters):
    """Improved migration logic with diversity"""
    current_step = parameters['current_step']
    
    edge_servers = [es for es in EdgeServer.all() if es is not None]
    if not edge_servers:
        return

    # Build state vectors
    state_vectors = []
    for es in edge_servers:
        state_vectors.extend([
            es.cpu,
            es.memory,
            es.disk,
            es.dynamic_power()
        ])
    
    # Get actions for all services
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
    return model.schedule.steps == 1000

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
        state_dim=state_dim,
        action_dim=len(valid_servers),
        hidden_dim=128,
        lr=0.0005,
        gamma=0.97,
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
    
    # Create subplots with your specified structure
    num_subplots = len(edge_server_ids) + 1
    fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 5*num_subplots), sharex=True)

    # Plot individual server power consumption with clean lines
    for i, server_id in enumerate(edge_server_ids):
        server_data = df[df['Instance ID'] == server_id]
        
        # Use simple line plot without markers
        axes[i].plot(server_data['Time Step'], 
                    server_data['Power Consumption'],
                    color='blue',
                    linewidth=1.5,
                    label=f"Edge Server {server_id}")
        
        axes[i].set_title(f"Edge Server {server_id} Power Consumption", fontsize=12)
        axes[i].set_ylabel("Power (W)", fontsize=10)
        axes[i].grid(True, alpha=0.2)
        axes[i].legend(loc='upper right')

    # Total power subplot with clean styling
    axes[-1].plot(range(len(power_history)), power_history,
                color='darkgreen',
                linewidth=2,
                label='Total System Power')
    
    axes[-1].set_title("Total Power Consumption Over Time", fontsize=12)
    axes[-1].set_xlabel("Time Step", fontsize=10)
    axes[-1].set_ylabel("Total Power (W)", fontsize=10)
    axes[-1].grid(True, alpha=0.2)
    axes[-1].legend()

    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    plt.savefig("Qlearning_power_subplots.png", dpi=300, bbox_inches='tight')
    plt.show()