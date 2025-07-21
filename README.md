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

<hr/>
<img style="max-height:512px" src="https://github.com/cekkr/learnit-r8/blob/main/assets/gem_graph.png?raw=true"/>
<i>Learning rate average approach (graph confused but happy to exist)</i>
<hr/>

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

# Deeply deep delving-doo

## How It Works: The `learnit::r8` Philosophy

The core idea behind `learnit::r8` is to treat the training process as a dynamic system rather than a static one. Instead of just shuffling your dataset and following a blind learning rate schedule, the scheduler actively observes the model's performance on **individual data points** and adjusts its strategy in real-time. 🧠

This process can be broken down into a few key phases and concepts.

### The Core Interaction Loop

Your interaction with the scheduler is designed to be simple and fit naturally into any training script. You don't tell the scheduler what data to use; you **ask** it for the next batch.

The loop always follows this pattern:

1.  **Ask for a batch**: The `scheduler.batch_iterator()` yields the indices for the next batch and the optimal learning rate (`lr`) for the current step.
2.  **Train your model**: You use these indices to fetch your data and run a single training step with the provided `lr`.
3.  **Report back**: You call `scheduler.update_with_loss(loss)` to tell the scheduler the outcome. This single call is where all the magic happens: the scheduler updates the UI, logs the performance of the data in that batch, and prepares for the next step.

<!-- end list -->

```python
# The standard training loop with learnit_r8
for epoch in scheduler.epoch_iterator():
    for batch_indices, lr in scheduler.batch_iterator():
        # 1. You get indices and the learning rate from the scheduler
        
        # 2. You perform your training step
        loss = my_train_step(batch_indices, lr) 
        
        # 3. You report the loss back. The scheduler handles the rest.
        scheduler.update_with_loss(loss)
```

-----

### Phase 1: The Cataloging Epoch (Epoch 1) 🧐

The very first epoch is special. Its primary goal is **data profiling**. The scheduler iterates through your dataset, often randomly, to ensure it sees every sample at least once. For each batch, it records the resulting loss and assigns an initial "difficulty score" (an exponential moving average of loss) to each data point involved.

By the end of Epoch 1, the scheduler has built a complete **loss profile** of your dataset for the initial state of your model. It now knows which data points are "easy" (low loss) and which are "hard" or "exploratory" (high loss).

-----

### Phase 2: Intelligent Batch Construction 🎯

From the second epoch onwards, the scheduler stops using simple random shuffling. Instead, it becomes a "smart" sampler.

1.  **Categorization**: It sorts all data points by their current average loss and divides them into four quantiles (or "quadrants").

      * **Q1**: The 25% of data with the lowest loss (**stable data**).
      * **Q2 & Q3**: The 50% of data with medium loss.
      * **Q4**: The 25% of data with the highest loss (**challenging or noisy data**).

2.  **Balanced Sampling**: When constructing a new batch, it doesn't just pick randomly. It intentionally pulls a proportional number of samples **from each quadrant**.

This is the key to its effectiveness. Each batch contains a carefully curated mix of:

  * ✅ **Easy samples (Q1)** to help stabilize the gradients and reinforce learning.
  * ▶️ **Medium samples (Q2, Q3)** to form the bulk of the learning process.
  * 🔼 **Hard samples (Q4)** to challenge the model, prevent it from getting lazy, and help it explore the boundaries of its understanding.

To prevent overfitting on a few specific "hard" examples, the scheduler also tracks a `usage_count` for each data point, prioritizing those that have been seen less frequently within each quadrant.

-----

### Stability & Automated Recovery 🛡️

Machine learning training can be unstable. `learnit::r8` has a built-in safety net.

  * **Gradient Explosion Detection**: If `update_with_loss()` receives a `NaN` or an extremely high loss value, it assumes a gradient explosion has occurred.
  * **Instant Recovery**: This is where the **CPU backup** comes in. During training, you can periodically save a known-good copy of your model's state to RAM (which is much faster than saving to disk).

<!-- end list -->

```python
# Inside your training loop, maybe every 50 steps
if scheduler.global_step % 50 == 0:
    # In PyTorch, this would be model.state_dict().to('cpu')
    current_model_state_on_cpu = get_current_model_weights_on_cpu()
    scheduler.update_cpu_backup(current_model_state_on_cpu)
```

If an explosion happens, the scheduler automatically triggers your `load_callback` with this in-memory backup, instantly restoring the model to its last stable state and allowing training to continue, often with a temporarily reduced learning rate. This turns a potentially run-ending crash into a small, self-corrected hiccup.

### Estimating Single Data Point Loss: The Math

A key challenge is that during training, you get a **single loss value for an entire batch** of data. How can we possibly know the specific contribution of each individual sample to that loss? The system is mathematically underdetermined for a single step.

The solution is to estimate it statistically over time. `learnit::r8` uses a method based on a corrective **Exponential Moving Average (EMA)**. The core idea is that while we can't know the true loss of a sample from one batch, we can refine our estimate for it each time it's used.

Here is the process for each training step:

1.  **Predict**: Before the update, the scheduler looks at the samples in the batch it just provided. Based on their historical performance, it calculates an **expected loss** for the batch by averaging the current `avg_loss` of all samples within it.

2.  **Observe**: It receives the **actual loss** (`batch_loss`) that you provide from your model's training step.

3.  **Correct**: It calculates the difference between the actual and expected loss. We call this the **"surprise" ($\\delta$)**, as it represents how much better or worse the batch performed than anticipated.

    $$
    $$$$\\delta = \\text{actual\_loss} - \\text{expected\_loss}

    $$
    $$$$
    $$
4.  **Update**: The scheduler updates the `avg_loss` for *every sample* in that batch by adding a small fraction of the "surprise" to their previous average.

    The mathematical update rule for a sample $i$ that was in the batch is:

    $$
    $$$$\\text{AvgLoss}*{\\text{new}}(i) = \\text{AvgLoss}*{\\text{old}}(i) + \\alpha \\cdot \\delta

    $$
    $$$$Where $\\alpha$ is a small smoothing factor (e.g., `0.1`), which acts like a learning rate for the loss estimation itself.

**Why is this effective?** This method is more robust than simply averaging the batch loss. If a batch performs surprisingly well (a negative $\\delta$), all samples in it get a small credit, nudging their estimated loss down. If it performs poorly (a positive $\\delta$), they all share a bit of the blame. Over many epochs, as samples appear in different batches with different companions, these small, iterative corrections converge towards a stable and accurate estimate of each sample's true "difficulty."

<hr>
Vibe coded by Riccardo Cecchini & Gemini ★★★★
