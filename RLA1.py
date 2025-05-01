"""
RL Master Course Assignment Solution
Q-Learning: Tabular & Deep for CartPole environment
"""

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import os
from tqdm import tqdm
import json

# Set random seeds for reproducibility
RANDOM_SEED = 42

def set_seeds(seed):
    """Set all random seeds to a fixed value"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

set_seeds(RANDOM_SEED)

# Define the DQN model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=[16]):
        super(DQN, self).__init__()

        # Build the network architecture based on hidden_sizes parameter
        layers = []
        prev_size = input_dim

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Experience replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = self.Transition(state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# DQN Agent with possible use of Target Network and Experience Replay
class DQNAgent:
    def __init__(self, state_dim, action_dim,
                 learning_rate=0.001,
                 gamma=0.99,  # Discount factor
                 epsilon_start=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.995, # Epsilon decay factor
                 hidden_sizes=[32],
                 buffer_size=10000, # Replay buffer size
                 batch_size=64,
                 update_target_every=100, # Update target network every 100 episodes
                 update_every=1, # Update model every step
                 use_target_network=False, 
                 use_experience_replay=False):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.update_every = update_every
        self.use_target_network = use_target_network
        self.use_experience_replay = use_experience_replay
        self.hidden_sizes = hidden_sizes

        # Initialize Q network and target network
        self.q_network = DQN(state_dim, action_dim, hidden_sizes)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # Initialize target network if used
        if use_target_network:
            self.target_network = DQN(state_dim, action_dim, hidden_sizes)
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Initialize replay buffer if used
        if use_experience_replay:
            self.replay_buffer = ReplayBuffer(buffer_size)

        self.train_step = 0

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            # Explore: random action
            return random.randint(0, self.action_dim - 1)
        else:
            # Exploit: best action according to Q-values
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                return torch.argmax(q_values).item()

    def update_epsilon(self):
        # Update epsilon by multiplying it with the decay factor
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def store_transition(self, state, action, reward, next_state, done):
        if self.use_experience_replay:
            self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        self.train_step += 1

        # Skip update if it's not time yet
        if self.train_step % self.update_every != 0:
            return

        # Skip update if not enough samples in replay buffer
        if self.use_experience_replay and len(self.replay_buffer) < self.batch_size:
            return

        # Get the experiences to learn from
        if self.use_experience_replay:
            # Sample a batch from replay buffer
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

            # Convert to tensors
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones)

            # Get current Q values
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # Get next Q values from target network or policy network
            if self.use_target_network:
                next_q_values = self.target_network(next_states).max(1)[0]
            else:
                next_q_values = self.q_network(next_states).max(1)[0]

            # Compute target Q values
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        else:
            # Single experience update (no replay buffer)
            # This is just a placeholder since we need to get single experiences
            # from somewhere else (training loop), and it's not applicable here
            return

        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values.detach())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network if needed
        if self.use_target_network and self.train_step % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def single_update(self, state, action, reward, next_state, done):
        """Update function for non-experience replay case"""
        # Convert to tensors
        state = torch.FloatTensor(state).unsqueeze(0)
        action = torch.LongTensor([action])
        reward = torch.FloatTensor([reward])
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        done = torch.FloatTensor([done])

        # Get current Q value
        current_q_value = self.q_network(state).gather(1, action.unsqueeze(1)).squeeze(1)

        # Get next Q value
        if self.use_target_network:
            next_q_value = self.target_network(next_state).max(1)[0]
        else:
            next_q_value = self.q_network(next_state).max(1)[0]
        
        # Compute target Q value
        target_q_value = reward + (1 - done) * self.gamma * next_q_value

        # Compute loss
        loss = self.loss_fn(current_q_value, target_q_value.detach())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network if needed
        if self.use_target_network and self.train_step % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.train_step += 1

def run_experiment(agent_config, env_name='CartPole-v1', max_steps=100000, eval_interval=500, seed=RANDOM_SEED):
    """Run a complete training experiment with the given agent configuration"""

    # Set seed for this experiment``
    set_seeds(seed)

    # Create environment
    env = gym.make(env_name)

    # Reset environment to get state dimension
    observation, _ = env.reset(seed=seed)
    state_dim = len(observation)
    action_dim = env.action_space.n

    # Create agent with the provided configuration
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        **agent_config
    )

    # Track training progress
    total_steps = 0
    episode = 0
    returns = []
    evaluation_returns = []
    evaluation_steps = []  # Track evaluation steps

    # Main training loop
    with tqdm(total=max_steps, desc="Training") as pbar:
        while total_steps < max_steps:
            state, _ = env.reset()
            done = False
            truncated = False
            episode_return = 0
            episode_steps = 0

            # Episode loop
            while not (done or truncated) and total_steps < max_steps:
                # Select and take action
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated  # Fix for the bool8 issue

                # Store transition
                agent.store_transition(state, action, reward, next_state, done or truncated)

                # Update agent (for single experience case or with replay buffer)
                if agent.use_experience_replay:
                    agent.update()
                else:
                    agent.single_update(state, action, reward, next_state, done or truncated)

                # Update state
                state = next_state

                # Update counters
                episode_return += reward
                episode_steps += 1
                total_steps += 1
                pbar.update(1)

                # Evaluate periodically
                if total_steps % eval_interval == 0:
                    eval_return = evaluate_agent(agent, env_name, num_episodes=30)
                    evaluation_returns.append(eval_return)
                    evaluation_steps.append(total_steps)  # Append current step
                    print(f"Steps: {total_steps}, Episode: {episode}, Evaluation Return: {eval_return:.2f}", flush=True)
                    pbar.set_postfix(eval_return=f"{eval_return:.2f}")

            # Update epsilon at the end of episode
            agent.update_epsilon()

            # Track episode return
            returns.append(episode_return)

            # Update episode counter
            episode += 1

    # Final evaluation
    final_eval_return = evaluate_agent(agent, env_name, num_episodes=30)

    # Close environment
    env.close()

    return {
        'returns': returns,
        'evaluation_returns': evaluation_returns,
        'evaluation_steps': evaluation_steps,  # Include evaluation steps
        'final_eval_return': final_eval_return,
        'config': agent_config
    }

def evaluate_agent(agent, env_name, num_episodes=30, max_episode_length=500, seed_offset=0):
    """Evaluate the agent performance without exploration"""
    eval_env = gym.make(env_name)
    eval_returns = []

    for episode_idx in range(num_episodes):
        # Use different seed for each evaluation episode
        state, _ = eval_env.reset(seed=RANDOM_SEED + seed_offset + episode_idx)
        done = False
        truncated = False
        episode_return = 0
        episode_length = 0

        # Limit maximum steps
        while not (done or truncated) and episode_length < max_episode_length:
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated  # Fix for the bool8 issue
            episode_return += reward
            state = next_state
            episode_length += 1

        eval_returns.append(episode_return)

    eval_env.close()

    # Return the average return
    return np.mean(eval_returns)

def run_experiment_with_repetitions(agent_config, num_repetitions=5, **kwargs):
    """Run experiment multiple times and average results"""
    all_results = []

    for rep in range(num_repetitions):
        print(f"Running repetition {rep+1}/{num_repetitions}")
        result = run_experiment(agent_config, seed=RANDOM_SEED+rep, **kwargs)
        all_results.append(result)

    # Find the minimum length of returns arrays
    min_returns_len = min(len(r['returns']) for r in all_results)

    # Truncate all returns arrays to the same length before averaging
    truncated_returns = [r['returns'][:min_returns_len] for r in all_results]

    # Aggregate results
    avg_result = {
        'returns': np.mean(truncated_returns, axis=0).tolist(),
        'evaluation_returns': np.mean([r['evaluation_returns'] for r in all_results], axis=0).tolist(),
        'evaluation_steps': all_results[0]['evaluation_steps'],  # Same for all repetitions
        'final_eval_return': np.mean([r['final_eval_return'] for r in all_results]),
        'config': agent_config
    }

    return avg_result

def run_ablation_and_plot_immediately(save_dir='results'):
    """运行每种超参数消融研究并立即绘制图表"""
    os.makedirs(save_dir, exist_ok=True)

    # 基础配置
    base_config = {
        'learning_rate': 0.001,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'hidden_sizes': [16],
        'update_every': 1,
        'use_target_network': False,
        'use_experience_replay': False
    }

    # hp settings
    learning_rates = [0.0005, 0.001, 0.01]
    hidden_sizes_options = [[16], [32], [64]]
    #update_every_options = [1, 4, 8]
    gamma_options = [0.95, 0.99, 0.999]
    epsilon_decay_options = [0.99, 0.995, 0.999]

    max_steps = 100000  

    # test learning rate and plot immediately
    print("Running learning rate ablation study...")
    lr_results = {}
    for lr in learning_rates:
        config = base_config.copy()
        config['learning_rate'] = lr
        lr_results[f'lr_{lr}'] = run_experiment_with_repetitions(config, max_steps=max_steps)

    # plot immediately
    plot_single_parameter_ablation(lr_results, 'Learning Rate Ablation', save_dir, 'learning_rate')

    # test hidden sizes and plot immediately
    print("Running hidden sizes ablation study...")
    hidden_results = {}
    for hidden_sizes in hidden_sizes_options:
        config = base_config.copy()
        config['hidden_sizes'] = hidden_sizes
        hidden_results[f'hidden_{"-".join(map(str, hidden_sizes))}'] = run_experiment_with_repetitions(config, max_steps=max_steps)

    
    plot_single_parameter_ablation(hidden_results, 'Network Architecture Ablation', save_dir, 'network_architecture')

    # test update frequency and plot immediately
    print("Running update frequency ablation study...")
    gamma_results = {}
    for gamma in gamma_options:
        config = base_config.copy()
        config['gamma'] = gamma  # Fix assignment here
        gamma_results[f'gamma_{gamma}'] = run_experiment_with_repetitions(config, max_steps=max_steps)
    
    # plot immediately
    plot_single_parameter_ablation(gamma_results, 'Discount Factor Ablation', save_dir, 'discount_factor')

    # test epsilon decay and plot immediately
    print("Running epsilon decay ablation study...")
    eps_results = {}
    for epsilon_decay in epsilon_decay_options:
        config = base_config.copy()
        config['epsilon_decay'] = epsilon_decay
        eps_results[f'eps_decay_{epsilon_decay}'] = run_experiment_with_repetitions(config, max_steps=max_steps)


    plot_single_parameter_ablation(eps_results, 'Exploration (Epsilon Decay) Ablation', save_dir, 'epsilon_decay')

    # emerage all results
    all_results = {}
    all_results.update(lr_results)
    all_results.update(hidden_results)
    all_results.update(gamma_results)
    #all_results.update(update_results)
    all_results.update(eps_results)

    return all_results

def plot_single_parameter_ablation(results, title, save_dir, file_prefix):
    """plot results from single parameter ablation study"""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(title)

    # evaluation returns
    for name, result in results.items():
        ax.plot(result['evaluation_steps'], result['evaluation_returns'], label=name)

    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Average Evaluation Return')
    ax.legend()
    ax.set_title('Evaluation Performance')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/{file_prefix}_ablation.png')
    plt.close()

    # save results as JSON
    for name, result in results.items():
        result_copy = result.copy()
        with open(f'{save_dir}/{name}_results.json', 'w') as f:
            json.dump(result_copy, f, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

def ablation_study_components():
    """Test different combinations of Target Network and Experience Replay"""
    # Define the best hyperparameters found in the first ablation study
    best_config = {
        'learning_rate': 0.001,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'hidden_sizes': [32],
        'update_every': 1,
        'use_target_network': False,
        'use_experience_replay': False
    }

    results = {}

    # Using 200,000 steps for component ablation study
    # This is a balance between meaningful results and reasonable run time
    # Full 1M steps would be ideal but very time-consuming
    max_steps = 100000

    # Naive - No Target Network, No Experience Replay
    config = best_config.copy()
    config['use_target_network'] = False
    config['use_experience_replay'] = False
    results['naive'] = run_experiment_with_repetitions(config, max_steps=max_steps)

    # Only Target Network
    config = best_config.copy()
    config['use_target_network'] = True
    config['use_experience_replay'] = False
    results['target_network'] = run_experiment_with_repetitions(config, max_steps=max_steps)

    # Only Experience Replay
    config = best_config.copy()
    config['use_target_network'] = False
    config['use_experience_replay'] = True
    results['experience_replay'] = run_experiment_with_repetitions(config, max_steps=max_steps)

    # Both Target Network and Experience Replay
    config = best_config.copy()
    config['use_target_network'] = True
    config['use_experience_replay'] = True
    results['target_network_experience_replay'] = run_experiment_with_repetitions(config, max_steps=max_steps)

    return results

def plot_hyperparameter_ablation(results, save_dir='results'):
    """Plot results from hyperparameter ablation study"""
    os.makedirs(save_dir, exist_ok=True)

    # Group results by hyperparameter type
    lr_results = {k: v for k, v in results.items() if k.startswith('lr_')}
    hidden_results = {k: v for k, v in results.items() if k.startswith('hidden_')}
    #update_results = {k: v for k, v in results.items() if k.startswith('update_every_')}
    gamma_results = {k: v for k, v in results.items() if k.startswith('gamma_')}
    eps_results = {k: v for k, v in results.items() if k.startswith('eps_decay_')}

    # Create plots with evaluation returns vs timesteps

    # Plot learning rate ablation
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Learning Rate Ablation')

    # Evaluation returns
    for name, result in lr_results.items():
        ax.plot(result['evaluation_steps'], result['evaluation_returns'], label=name)

    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Average Evaluation Return')
    ax.legend()
    ax.set_title('Evaluation Performance')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/learning_rate_ablation.png')
    plt.close()

    # Plot hidden sizes ablation
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Network Architecture Ablation')

    # Evaluation returns
    for name, result in hidden_results.items():
        ax.plot(result['evaluation_steps'], result['evaluation_returns'], label=name)

    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Average Evaluation Return')
    ax.legend()
    ax.set_title('Evaluation Performance')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/network_architecture_ablation.png')
    plt.close()

    # Plot update frequency ablation
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Discount Factor Ablation')

    # Evaluation returns
    for name, result in gamma_results.items():
        ax.plot(result['evaluation_steps'], result['evaluation_returns'], label=name)

    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Average Evaluation Return')
    ax.legend()
    ax.set_title('Evaluation Performance')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/discount_factor_ablation .png')
    plt.close()

    # Plot epsilon decay ablation
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Exploration (Epsilon Decay) Ablation')

    # Evaluation returns
    for name, result in eps_results.items():
        ax.plot(result['evaluation_steps'], result['evaluation_returns'], label=name)

    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Average Evaluation Return')
    ax.legend()
    ax.set_title('Evaluation Performance')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/epsilon_decay_ablation.png')
    plt.close()

    # Save results as JSON
    for name, result in results.items():
        # Convert numpy arrays to lists for JSON serialization
        result_copy = result.copy()

        with open(f'{save_dir}/{name}_results.json', 'w') as f:
            json.dump(result_copy, f, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

def plot_component_ablation(results, save_dir='results'):
    """Plot results from component ablation study (TN, ER) using timesteps on x-axis"""
    os.makedirs(save_dir, exist_ok=True)

    # Create plots with evaluation returns vs timesteps

    # Plot for Naive
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Naive (No Target Network, No Experience Replay)')
    ax.plot(results['naive']['evaluation_steps'], results['naive']['evaluation_returns'], label='Naive')
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Average Evaluation Return')
    ax.legend()
    ax.set_title('Evaluation Performance')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/naive_component_ablation.png')
    plt.close()

    # Plot for Only Target Network
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Only Target Network')
    ax.plot(results['target_network']['evaluation_steps'], results['target_network']['evaluation_returns'], label='Only Target Network')
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Average Evaluation Return')
    ax.legend()
    ax.set_title('Evaluation Performance')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/target_network_component_ablation.png')
    plt.close()

    # Plot for Only Experience Replay
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Only Experience Replay')
    ax.plot(results['experience_replay']['evaluation_steps'], results['experience_replay']['evaluation_returns'], label='Only Experience Replay')
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Average Evaluation Return')
    ax.legend()
    ax.set_title('Evaluation Performance')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/experience_replay_component_ablation.png')
    plt.close()

    # Plot for Target Network & Experience Replay
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Target Network & Experience Replay')
    ax.plot(results['target_network_experience_replay']['evaluation_steps'], results['target_network_experience_replay']['evaluation_returns'], label='Target Network & Experience Replay')
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Average Evaluation Return')
    ax.legend()
    ax.set_title('Evaluation Performance')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/target_network_experience_replay_component_ablation.png')
    plt.close()

    # Save results as JSON
    for name, result in results.items():
        # Convert numpy arrays to lists for JSON serialization
        result_copy = result.copy()
        with open(f'{save_dir}/{name}_results.json', 'w') as f:
            json.dump(result_copy, f, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

def main():
    # create result content
    os.makedirs('results', exist_ok=True)

    # run hyperparameter ablation studies
    print("Running hyperparameter ablation studies with immediate plotting...")
    #hyperparameter_results = run_ablation_and_plot_immediately()

    # run component ablation study
    print("Running component ablation study (TN, ER)...")
    component_results = ablation_study_components()
    plot_component_ablation(component_results)

    print("All experiments completed. Results saved to 'results' directory.")

if __name__ == "__main__":
    main()