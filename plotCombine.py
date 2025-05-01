import json
import matplotlib.pyplot as plt

# Load data from JSON files
def load_results(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

# Plot results
def plot_results():
    files = {
        "Naive": r"e:\Leiden\Deep Reignforcement Learning\A1\Final results\Colab epsilon decay component\naive_results.json",
        "Experience Replay": r"e:\Leiden\Deep Reignforcement Learning\A1\Final results\Colab epsilon decay component\experience_replay_results.json",
        "Target Network": r"e:\Leiden\Deep Reignforcement Learning\A1\Final results\Colab epsilon decay component\target_network_results.json",
        "Target Network + Experience Replay": r"e:\Leiden\Deep Reignforcement Learning\A1\Final results\Colab epsilon decay component\target_network_experience_replay_results.json"
    }

    plt.figure(figsize=(10, 6))
    
    for label, filepath in files.items():
        data = load_results(filepath)
        plt.plot(data["evaluation_steps"], data["evaluation_returns"], label=label)

    plt.title("Evaluation Returns vs Steps")
    plt.xlabel("Steps")
    plt.ylabel("Evaluation Returns")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_results()
