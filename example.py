import time
import random
import os
import shutil

# Import the main class from the library
from learnit_r8 import R8Scheduler

# --- Mock Components for Demonstration ---

# This simulates your model's state (e.g., weights)
mock_model_state_dict = {"weights": [random.random() for _ in range(10)]}

def my_train_step(batch_indices, lr):
    """A dummy training function."""
    # In a real application (PyTorch/TensorFlow), you would:
    # 1. Get data for batch_indices
    # 2. Move data to GPU
    # 3. optimizer.zero_grad()
    # 4. output = model(data)
    # 5. loss = criterion(output, target)
    # 6. loss.backward()
    # 7. optimizer.step()
    # 8. return loss.item()

    # Simulate work
    time.sleep(0.01)

    # Simulate a loss that decreases as training progresses and is sensitive to LR
    # A very high LR will give a higher (worse) loss
    base_loss = 1.0 / (lr * 100 + 1) if lr > 0 else 1.0
    noise = random.uniform(-0.1, 0.1) # Add some randomness
    final_loss = base_loss + noise
    
    # Simulate a random gradient explosion
    if random.random() < 0.005: # 0.5% chance
        return float('nan')
        
    return abs(final_loss)

def save_model_callback(filepath):
    """A dummy callback to save the model state."""
    # In PyTorch: torch.save(model.state_dict(), filepath)
    global mock_model_state_dict
    print(f"\n[Callback] Saving mock model with weights[0]={mock_model_state_dict['weights'][0]:.4f} to {filepath}...")
    # Here we would serialize the state_dict
    time.sleep(0.1) # Simulate disk I/O
    return True

def load_model_callback(source, from_ram=False):
    """A dummy callback to load a model state."""
    global mock_model_state_dict
    if from_ram:
        # source is the actual state_dict from CPU RAM
        print(f"\n[Callback] Loading model state directly from RAM backup...")
        mock_model_state_dict = source
    else:
        # source is a filepath
        print(f"\n[Callback] Loading mock model from {source}...")
        # Here we would deserialize
    time.sleep(0.1) # Simulate disk I/O
    return mock_model_state_dict

def main():
    """Main function to run the training demo."""
    # Clean up previous checkpoints
    if os.path.exists('./checkpoints'):
        shutil.rmtree('./checkpoints')

    # 1. Instantiate the scheduler with all configurations
    scheduler = R8Scheduler(
        num_samples=1000,
        num_epochs=10,
        batch_size=32,
        initial_lr=0.01,
        scheduler_mode='cosine', # 'linear' or 'cosine'
        use_curses=True,
        save_callback=save_model_callback,
        load_callback=load_model_callback,
        checkpoint_dir='./checkpoints'
    )

    print("Starting training with learnit::r8 scheduler...")
    time.sleep(2)

    # 2. The main training loop
    try:
        # The outer loop handles epochs
        for epoch in scheduler.epoch_iterator():
            
            # The inner loop provides batches and the correct LR for that step
            for batch_indices, lr in scheduler.batch_iterator():
                
                # Execute your training step
                loss = my_train_step(batch_indices, lr)
                
                # Provide the resulting loss back to the scheduler
                # It will handle all internal updates (UI, LR, data profiling)
                scheduler.update_with_loss(loss)

                # Optional: Periodically update the CPU RAM backup with the current model state
                # This should be a stable state, so maybe after a few steps with good loss
                if scheduler.global_step % 50 == 0:
                    # In PyTorch: scheduler.update_cpu_backup(model.state_dict().to('cpu'))
                    scheduler.update_cpu_backup(mock_model_state_dict.copy())

            # Optional: Save a checkpoint at the end of each epoch
            scheduler.save_checkpoint()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        # 3. Ensure resources are cleaned up
        scheduler.close()
        print("\nScheduler UI closed. Training finished.")

if __name__ == "__main__":
    main()