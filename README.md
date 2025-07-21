# learnit::r8 - An Intelligent ML Training Scheduler

`learnit::r8` is a Python library designed to provide an advanced, data-aware scheduling and management system for machine learning training loops. It operates generically through a callback system but can be easily integrated with popular frameworks like PyTorch or TensorFlow.

The name is stylized as `learnit::r8`, but the Python package name is `learnit_r8`.

## Core Features

-   **Dynamic Learning Rate Scheduling**: Implements linear and cosine schedulers with a warm-up phase to find optimal learning rates.
-   **Data-Aware Batching**: Intelligently selects data for each batch by categorizing samples based on their historical training loss. This ensures a balanced mix of "easy" and "hard" examples, promoting stable and efficient training.
-   **Live Terminal UI**: Uses `curses` to display a real-time dashboard with progress bars, loss metrics, and learning rates directly in your terminal.
-   **Automated Loss Cataloging**: In the first epoch, it profiles each data point to establish a baseline loss, which informs the subsequent batching strategy.
-   **Stabilization & Anti-Overfitting**:
    -   Uses low-loss ("stable") data to restabilize training if loss spikes occur.
    -   Considers the usage count of each sample to prevent overfitting on specific examples.
-   **Checkpointing & Crash Recovery**:
    -   Callback-driven system for saving and loading model states.
    -   Features a CPU RAM backup mechanism to instantly recover from gradient explosions without reloading from disk.

## Installation

```bash
pip install numpy
# On Linux/macOS, curses is standard. On Windows:
pip install windows-curses
```

Then, place the `learnit_r8` folder in your project directory.

## Quick Start

See the `example.py` file for a complete, runnable demonstration.

```python
import time
import random
from learnit_r8 import R8Scheduler

# 1. Define your training components
def my_train_step(batch_indices, lr):
    # Your actual model training logic goes here
    # print(f"Training on {len(batch_indices)} samples with LR: {lr:.6f}")
    time.sleep(0.05)
    # Simulate a loss that depends on the data and learning rate
    return 1.0 / (lr * 100 + 1) + random.uniform(-0.1, 0.1)

def save_model_callback(state_dict, filepath):
    # In a real scenario, you would save model weights, e.g., torch.save(...)
    print(f"\n[Callback] Saving model state to {filepath}...")
    return True

def load_model_callback(filepath):
    # In a real scenario, you would load model weights, e.g., model.load_state_dict(...)
    print(f"\n[Callback] Loading model state from {filepath}...")
    return {"key": "mock_model_state"} # Return the loaded state

# 2. Setup the Scheduler
scheduler = R8Scheduler(
    num_samples=1000,
    num_epochs=10,
    batch_size=32,
    initial_lr=0.01,
    scheduler_mode='cosine',
    save_callback=save_model_callback,
    load_callback=load_model_callback,
    use_curses=True
)

# 3. Run the training loop
try:
    for epoch in scheduler.epoch_iterator():
        for batch_indices, lr in scheduler.batch_iterator():
            # Perform a training step
            loss = my_train_step(batch_indices, lr)
            
            # Update the scheduler with the result
            scheduler.update_with_loss(loss)

finally:
    # Ensure curses UI is closed properly
    scheduler.close()

print("\nTraining complete!")
```

Vibe coded.