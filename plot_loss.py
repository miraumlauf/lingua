import re
import matplotlib.pyplot as plt

# Function to parse the train.log file
def parse_train_log(file_path):
    steps = []
    losses = []
    grads = []
    lrs = []
    
    # Regular expression to extract step, loss, grad, and learning rate
    log_pattern = re.compile(r'step:\s+(\d+).*?loss:\s+([\d.]+).*?grad:\s+([\deE.+-]+).*?lr:\s+([\deE.+-]+)')
    
    with open(file_path, 'r') as file:
        for line in file:
            match = log_pattern.search(line)
            if match:
                steps.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                grads.append(float(match.group(3)))
                lrs.append(float(match.group(4)))
    
    return steps, losses, grads, lrs

# Plotting the loss, gradient, and learning rate
def plot_metrics(steps, losses, grads, lrs, output_file):
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot loss on the first y-axis
    ax1.plot(steps, losses, label="Loss", color="blue", linewidth=2)
    ax1.set_xlabel("Steps", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12, color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.grid(True, linestyle="--", alpha=0.7)
    
    # Create a second y-axis for gradient
    ax2 = ax1.twinx()
    ax2.plot(steps, grads, label="Gradient", color="green", linewidth=2)
    ax2.set_ylabel("Gradient", fontsize=12, color="green")
    ax2.tick_params(axis='y', labelcolor="green")

    # Create a third y-axis for learning rate (scaled and aligned with ax2)
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))  # Offset the third y-axis
    ax3.plot(steps, lrs, label="Learning Rate", color="red", linewidth=2)
    ax3.set_ylabel("Learning Rate", fontsize=12, color="red")
    ax3.tick_params(axis='y', labelcolor="red")

    # Add legends for clarity
    ax1.legend(loc="upper left", fontsize=10)
    ax2.legend(loc="upper center", fontsize=10)
    ax3.legend(loc="upper right", fontsize=10)

    plt.title("Dim=256/Batch=64 Experiment: LR=3e-3, lr min ratio")
    
    # Save the plot as a PNG file
    plt.savefig(output_file, bbox_inches="tight")
    print(f"Plot saved as '{output_file}'.")

# Main logic
if __name__ == "__main__":
    # Update the path to your train.log file here
    file_path = "./apps/mtp/llama_babylm_lr_min/train.log"
    output_file = "./metrics/metrics_dim_256_lr_min.png"  # The name of the output PNG file
    steps, losses, grads, lrs = parse_train_log(file_path)
    if steps and losses and grads and lrs:
        plot_metrics(steps, losses, grads, lrs, output_file)
    else:
        print("No valid data found in the log file.")
