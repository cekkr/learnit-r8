import numpy as np
import math
import time
import os
import copy
from .ui import CursesUI
from .utils import save_state_to_json, load_state_from_json

class R8Scheduler:
    """
    Manages the ML training lifecycle, including LR scheduling,
    data-aware batching, and state management.
    """

    def __init__(self,
                 num_samples: int,
                 batch_size: int,
                 initial_lr: float = 0.01,
                 num_epochs: int = None,
                 train_duration_seconds: int = None,
                 scheduler_mode: str = 'cosine',
                 warmup_proportion: float = 0.25,
                 use_curses: bool = True,
                 save_callback=None,
                 load_callback=None,
                 checkpoint_dir: str = './checkpoints'):
        
        if not num_epochs and not train_duration_seconds:
            raise ValueError("Must provide either 'num_epochs' or 'train_duration_seconds'.")

        # Configuration
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.initial_lr = initial_lr
        self.scheduler_mode = scheduler_mode
        self.warmup_proportion = warmup_proportion
        self.save_callback = save_callback
        self.load_callback = load_callback
        self.checkpoint_dir = checkpoint_dir
        self.train_duration_seconds = train_duration_seconds

        # State
        self.current_epoch = 0
        self.global_step = 0
        self.current_batch_indices = []
        self.steps_per_epoch = math.ceil(self.num_samples / self.batch_size)
        
        # Time-based or Epoch-based training
        self.end_time = None
        if self.train_duration_seconds:
            self.start_time = time.time()
            self.end_time = self.start_time + self.train_duration_seconds
            # For LR scheduling, we still need a concept of total steps.
            # We estimate it based on a large number of epochs.
            self.num_epochs = 1000 # A large virtual number for scheduling
        else:
            self.start_time = time.time() # Start time is useful for ETA regardless
            self.num_epochs = num_epochs

        self.total_steps = self.steps_per_epoch * self.num_epochs
        self.warmup_steps = int(self.total_steps * self.warmup_proportion)
        
        # Data & Loss Tracking
        self.sample_stats = {
            i: {'avg_loss': float('inf'), 'usage_count': 0}
            for i in range(self.num_samples)
        }
        self.all_indices = np.arange(self.num_samples)
        self.epoch_usage_count = {i: 0 for i in range(self.num_samples)}
        
        # LR and Model State
        self.lr = self.initial_lr / 2.0
        self.cpu_backup = None
        self.last_stable_loss = float('inf')

        # UI
        self.use_curses = use_curses
        self.ui = CursesUI() if self.use_curses else None

    @classmethod
    def resume_from_checkpoint(cls, checkpoint_path: str, save_callback, load_callback, **kwargs):
        """Creates a scheduler instance by loading state from a checkpoint."""
        state = load_state_from_json(checkpoint_path)
        if not state:
            raise FileNotFoundError(f"Could not load scheduler state from {checkpoint_path}")

        # Initialize scheduler with saved config, but allow overrides
        config = {
            'num_samples': len(state['sample_stats']),
            'batch_size': state.get('batch_size', 32),
            'initial_lr': state.get('initial_lr', 0.01),
            'scheduler_mode': state.get('scheduler_mode', 'cosine'),
            'save_callback': save_callback,
            'load_callback': load_callback,
            **kwargs
        }
        
        scheduler = cls(**config)
        
        # Restore state
        scheduler.current_epoch = state.get('current_epoch', 0)
        scheduler.global_step = state.get('global_step', 0)
        scheduler.sample_stats = state.get('sample_stats', {})
        scheduler.lr = state.get('lr', scheduler.initial_lr / 2.0)
        
        if scheduler.ui:
            scheduler.ui.log_message(f"Resumed training from {checkpoint_path}", color_pair=2)
            
        return scheduler

    def _get_next_batch_indices(self):
        """Intelligently selects indices for the next batch."""
        if self.current_epoch == 0: # Random exploration for cataloging
            available_indices = [i for i, count in self.epoch_usage_count.items() if count == 0]
            if not available_indices: # Reset if all used
                self.epoch_usage_count = {i: 0 for i in range(self.num_samples)}
                available_indices = list(self.all_indices)
            
            batch = np.random.choice(available_indices, min(self.batch_size, len(available_indices)), replace=False).tolist()

        else: # Smart batching with loss targeting
            valid_samples = {i: s['avg_loss'] for i, s in self.sample_stats.items() if s['avg_loss'] != float('inf')}
            if not valid_samples:
                # Fallback to random if no loss data is available yet
                return np.random.choice(self.all_indices, self.batch_size, replace=False).tolist()

            target_avg_loss = np.mean(list(valid_samples.values()))
            
            # Use quantiles to get a diverse pool of candidates
            sorted_indices = sorted(valid_samples.keys(), key=lambda i: valid_samples[i])
            quantiles = np.array_split(sorted_indices, 4)
            
            batch = []
            # Allow samples to be used up to 2 times per epoch to ensure enough candidates
            available_indices_in_epoch = [i for i, count in self.epoch_usage_count.items() if count < 2] 
            
            for q in quantiles:
                # Prioritize less-used samples
                candidates = [idx for idx in q if idx in available_indices_in_epoch]
                sorted_q = sorted(candidates, key=lambda i: self.epoch_usage_count[i])
                batch.extend(sorted_q[:self.batch_size // 4])
            
            # Adjust batch to meet target loss
            if batch: # Only adjust if a batch was successfully formed
                batch = self._adjust_batch_to_target_loss(batch, target_avg_loss, available_indices_in_epoch)

        # Ensure batch is not empty and has the correct size if possible
        while len(batch) < self.batch_size:
            remaining_pool = [i for i in self.all_indices if i not in batch]
            if not remaining_pool: break
            batch.append(np.random.choice(remaining_pool))

        for idx in batch:
            self.epoch_usage_count[idx] += 1
        
        return batch

    def _adjust_batch_to_target_loss(self, batch, target_loss, available_pool, tolerance=0.1, max_swaps=5):
        """Swaps samples in/out of a batch to better match the target average loss."""
        # Ensure there are valid losses to calculate the mean
        batch_with_loss = [i for i in batch if self.sample_stats[i]['avg_loss'] != float('inf')]
        if not batch_with_loss:
            return batch

        current_loss = np.mean([self.sample_stats[i]['avg_loss'] for i in batch_with_loss])
        
        for _ in range(max_swaps):
            if target_loss > 0 and abs(current_loss - target_loss) / target_loss < tolerance:
                break

            error = current_loss - target_loss
            
            # Exclude batch samples from the available pool for swapping
            swap_pool = [i for i in available_pool if i not in batch and self.sample_stats[i]['avg_loss'] != float('inf')]
            if not swap_pool: break

            if error > 0: # Batch is too hard, swap a high-loss sample for a low-loss one
                sample_to_remove = max(batch, key=lambda i: self.sample_stats[i]['avg_loss'])
                sample_to_add = min(swap_pool, key=lambda i: self.sample_stats[i]['avg_loss'])
            else: # Batch is too easy, swap a low-loss sample for a high-loss one
                sample_to_remove = min(batch, key=lambda i: self.sample_stats[i]['avg_loss'])
                sample_to_add = max(swap_pool, key=lambda i: self.sample_stats[i]['avg_loss'])

            if sample_to_remove in batch:
                batch.remove(sample_to_remove)
                batch.append(sample_to_add)
                current_loss = np.mean([self.sample_stats[i]['avg_loss'] for i in batch if self.sample_stats[i]['avg_loss'] != float('inf')])
            else:
                break # Sample to remove not found, stop swapping
            
        return batch

    def update_with_loss(self, batch_loss: float):
        """Update internal state after a training step using the 'predict and correct' method."""
        if math.isnan(batch_loss) or (self.last_stable_loss != float('inf') and batch_loss > 10 * self.last_stable_loss):
            self._handle_gradient_explosion()
            return
            
        alpha = 0.1 # Smoothing factor
        valid_samples = [idx for idx in self.current_batch_indices if self.sample_stats[idx]['avg_loss'] != float('inf')]
        
        expected_loss = np.mean([self.sample_stats[idx]['avg_loss'] for idx in valid_samples]) if valid_samples else batch_loss
        surprise = batch_loss - expected_loss
        
        for idx in self.current_batch_indices:
            stats = self.sample_stats[idx]
            stats['usage_count'] += 1
            if stats['avg_loss'] == float('inf'):
                stats['avg_loss'] = batch_loss
            else:
                stats['avg_loss'] += alpha * surprise

        if self.last_stable_loss == float('inf'):
            self.last_stable_loss = batch_loss
        else:
            self.last_stable_loss = (self.last_stable_loss * 0.9 + batch_loss * 0.1)

        self.global_step += 1
        self._update_lr()

        # --- COMPLETED IMPLEMENTATION HERE ---
        if self.ui:
            # Calculate step within the current epoch for display
            step_in_epoch = (self.global_step - 1) % self.steps_per_epoch + 1
            
            # Determine total steps based on training mode
            total_display_steps = self.total_steps
            if self.train_duration_seconds:
                # Estimate total steps based on current speed for a better ETA
                elapsed_time = time.time() - self.start_time
                if elapsed_time > 1: # Avoid division by zero
                    steps_per_second = self.global_step / elapsed_time
                    total_display_steps = int(steps_per_second * self.train_duration_seconds)

            self.ui.update(
                epoch=self.current_epoch + 1,
                total_epochs=self.num_epochs,
                step=step_in_epoch,
                total_steps_in_epoch=self.steps_per_epoch,
                global_step=self.global_step,
                total_global_steps=total_display_steps,
                loss=batch_loss,
                lr=self.lr
            )
        # --- END OF IMPLEMENTATION ---

    def find_optimal_lr(self, model_state, train_step_func, start_lr=1e-7, end_lr=1.0, steps=100):
        """Performs a learning rate range test to find an optimal starting LR."""
        original_model_state = copy.deepcopy(model_state)
        
        lrs = np.geomspace(start_lr, end_lr, steps)
        losses = []
        best_loss = float('inf')

        print(f"Running LR finder for {steps} steps...")
        for i, lr in enumerate(lrs):
            batch_indices = np.random.choice(self.all_indices, self.batch_size, replace=False).tolist()
            loss = train_step_func(batch_indices, lr)
            
            if i > 10 and (math.isnan(loss) or loss > 4 * best_loss):
                print("\nLoss exploded. Stopping test.")
                break 
            
            if loss < best_loss:
                best_loss = loss

            losses.append(loss)
            print(f"\rStep {i+1}/{steps} | LR: {lr:.8f} | Loss: {loss:.4f}", end="")
        
        print("\nLR finder test complete.")
        
        if self.load_callback:
             self.load_callback(original_model_state, from_ram=True)

        if len(losses) < 5: return self.initial_lr

        losses_smoothed = np.convolve(losses, np.ones(5)/5, mode='valid')
        min_loss_idx = np.argmin(losses_smoothed)
        
        # Suggest LR one order of magnitude before the minimum loss point
        optimal_idx = max(0, min_loss_idx - int(steps/10))
        return lrs[optimal_idx]

    def epoch_iterator(self):
        """An iterator that yields for each epoch or runs indefinitely for time-based training."""
        epoch_counter = self.current_epoch
        while True:
            if self.end_time and time.time() >= self.end_time:
                break
            if not self.end_time and epoch_counter >= self.num_epochs:
                break
            
            self.current_epoch = epoch_counter
            self.epoch_usage_count = {i: 0 for i in range(self.num_samples)} # Reset usage for new epoch
            yield epoch_counter
            epoch_counter += 1

    def batch_iterator(self):
        """An iterator that yields batch indices and learning rate, respects time limit."""
        steps_in_epoch = math.ceil(self.num_samples / self.batch_size)
        
        # In time-based mode, loop 'indefinitely' within an epoch
        max_steps = float('inf') if self.train_duration_seconds else steps_in_epoch
        
        step_counter = 0
        while step_counter < max_steps:
            if self.end_time and time.time() >= self.end_time:
                raise StopIteration
            if not self.train_duration_seconds and self.global_step >= self.total_steps:
                raise StopIteration
            
            self.current_batch_indices = self._get_next_batch_indices()
            yield self.current_batch_indices, self.lr
            step_counter += 1

    def load_checkpoint(self, filepath):
        """Loads the scheduler and model state from a checkpoint."""
        if not self.load_callback: return

        state = load_state_from_json(filepath)
        if state:
            self.current_epoch = state.get('current_epoch', 0)
            self.global_step = state.get('global_step', 0)
            self.sample_stats = state.get('sample_stats', self.sample_stats)
            
            model_path = filepath.replace('scheduler_', 'model_').replace('.json', '.bin')
            self.load_callback(model_path)
            if self.ui:
                self.ui.log_message(f"Checkpoint loaded from {filepath}", color_pair=2)

    def _update_lr(self):
        # In time-based mode, total_steps can be a rough estimate.
        # This schedule remains valid as it's based on progress through this estimate.
        if self.global_step < self.warmup_steps:
            self.lr = (self.initial_lr / 2) + (self.initial_lr / 2) * (self.global_step / self.warmup_steps)
        else:
            progress = (self.global_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            if self.scheduler_mode == 'cosine':
                self.lr = self.initial_lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
            else: # 'linear'
                self.lr = self.initial_lr * (1 - min(progress, 1.0))
        self.lr = max(self.lr, 1e-8)

    def _handle_gradient_explosion(self):
        if self.ui: self.ui.log_message("Gradient explosion detected! Attempting recovery...", color_pair=4)
        if self.cpu_backup and self.load_callback:
            self.load_callback(self.cpu_backup, from_ram=True)
            if self.ui: self.ui.log_message("Recovered model state from RAM backup.", color_pair=2)
            self.lr *= 0.5
        else:
            if self.ui: self.ui.log_message("No CPU backup found. Cannot recover automatically.", color_pair=4)
    
    def update_cpu_backup(self, model_state):
        self.cpu_backup = copy.deepcopy(model_state)
    
    def save_checkpoint(self, is_final=False):
        """Saves the scheduler and model state."""
        if not self.save_callback: 
            return
        
        state = {'current_epoch': self.current_epoch, 'global_step': self.global_step, 'sample_stats': self.sample_stats, 'initial_lr': self.initial_lr, 'scheduler_mode': self.scheduler_mode, 'batch_size': self.batch_size, 'lr': self.lr}
        suffix = 'final' if is_final else f'epoch_{self.current_epoch}_step_{self.global_step}'
        scheduler_path = os.path.join(self.checkpoint_dir, f'scheduler_{suffix}.json')
        model_path = os.path.join(self.checkpoint_dir, f'model_{suffix}.bin')
        
        save_state_to_json(state, scheduler_path)
        self.save_callback(model_path)

        if self.ui:
            self.ui.log_message(f"Checkpoint saved to {self.checkpoint_dir}", color_pair=2)
    
    def close(self):
        """Safely closes the Curses UI."""
        if self.ui: 
            self.ui.close()