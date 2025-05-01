# Deep Reinforcement Learning - RL_A0

This project contains implementations and experiments for deep reinforcement learning techniques, including Q-Learning and Deep Q-Networks (DQN). The experiments are conducted on the CartPole environment using various configurations and ablation studies.

## Project Structure

- **RLA1.py**: Main script containing the implementation of the DQN agent, training loop, and ablation studies.
- **plotCombine.py**: Script for combining and visualizing results from different experiments(As we plot the results for Naive - Only TN - Only ER - TN & ER in RLA1.py).


## Key Features

- **DQN Implementation**: Includes support for target networks and experience replay.
- **Ablation Studies**: Hyperparameter tuning and component analysis (e.g., target networks, experience replay).
- **Visualization**: Scripts for plotting evaluation results and comparing configurations.

## How to Run

1. Install dependencies:
   
   pip install -r requirements.txt
   
   Ensure you have Python 3.8+ installed.

2. Run the main script:
   
   python RLA1.py
   
   This will show the result of ablation study and Component Study, saved in PNG and JSON files independently.

3. Visualize results:
   
   python plotCombine.py
   
   **Note**: when combining the plot from JSON file, make sure to add the Correct location of the JSON files of the result for Naive - Only TN - Only ER - TN & ER. 

## Results

Results from experiments are saved in the `results` directory. Plots and JSON files summarize the performance of different configurations.

## Dependencies

- Python 3.8+
- NumPy
- PyTorch
- Gymnasium
- Matplotlib
- TQDM
