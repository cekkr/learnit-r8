import numpy as np
import math
import time
import os
from .ui import CursesUI
from .utils import save_state_to_json, load_state_from_json

class R8Scheduler:
    """
    Manages the ML training lifecycle, including LR scheduling,
    data-aware batching, and state management.
    """

    def __init__(self,
                 num_samples: int,
                 num_epochs: int,
                 batch_size: int,
                 initial_lr: float,
                 scheduler_mode: str = 'linear',
                 warmup_proportion: float = 0.25,
                 use_curses: bool = True,
                 save_callback=None,
                 load_callback=None,
                 checkpoint_dir: str = './checkpoints'):
        
        # Configuration
        self.num_samples = num_samples
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.initial_lr = initial_lr
        self.scheduler_mode = scheduler_mode
        self.warmup_proportion = warmup_proportion
        self.save_callback = save_callback
        self.load_callback = load_callback
        self.checkpoint_dir = checkpoint_dir

        # State
        self.current_epoch = 0
        self.global_step = 0
        self.current_batch_indices = []
        self.steps_per_epoch = math.ceil(self.num_samples / self.batch_size)
        self.total_steps = self.steps_per_epoch * self.num_epochs
        self.warmup_steps = int(self.total_steps * self.warmup_proportion)
        
        # Data & Loss Tracking
        self.sample_stats = {
            i: {'avg_loss': float('inf'), 'usage_count': 0, 'loss_history': []}
            for i in range(self.num_samples)
        }
        self.all_indices = np.arange(self.num_samples)
        self.unseen_indices_in_epoch = list(self.all_indices)
        
        # LR and Model State
        self.lr = self.initial_lr / 2.0  # Start at half LR
        self.cpu_backup = None
        self.last_stable_loss = float('inf')

        # UI
        self.use_curses = use_curses
        self.ui = CursesUI() if self.use_curses else None

    def _update_lr(self):
        """Updates the learning rate based on the current step."""
        if self.global_step < self.warmup_steps:
            # Linear warm-up from initial_lr/2 to initial_lr
            self.lr = (self.initial_lr / 2) + (self.initial_lr / 2) * (self.global_step / self.warmup_steps)
        else:
            # Cool-down phase
            progress = (self.global_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            if self.scheduler_mode == 'cosine':
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                self.lr = self.initial_lr * cosine_decay
            else: # 'linear'
                self.lr = self.initial_lr * (1 - progress)

        # Prevent LR from being exactly zero
        self.lr = max(self.lr, 1e-8)

    def _get_next_batch_indices(self):
        """Intelligently selects indices for the next batch."""
        # Epoch 1: Use random unseen data to catalog initial losses
        if self.current_epoch == 0:
            if not self.unseen_indices_in_epoch:
                 self.unseen_indices_in_epoch = list(self.all_indices) # Should not happen with ceil
            
            np.random.shuffle(self.unseen_indices_in_epoch)
            batch = self.unseen_indices_in_epoch[:self.batch_size]
            self.unseen_indices_in_epoch = self.unseen_indices_in_epoch[self.batch_size:]
            return batch

        # Subsequent epochs: Smart batching
        # 1. Classify samples by loss (filter out those with infinite loss)
        valid_samples = {i: s['avg_loss'] for i, s in self.sample_stats.items() if s['avg_loss'] != float('inf')}
        if not valid_samples: # Fallback if no valid losses yet
            return np.random.choice(self.all_indices, self.batch_size, replace=False).tolist()

        sorted_indices = sorted(valid_samples.keys(), key=lambda i: valid_samples[i])
        
        # 2. Divide into 4 quantiles (quadrants)
        num_quantiles = 4
        quantiles = np.array_split(sorted_indices, num_quantiles)
        
        batch = []
        samples_per_quantile = self.batch_size // num_quantiles
        
        # 3. Sample from each quantile, prioritizing less-used data
        for q in quantiles:
            if not len(q): continue
            # Sort quantile by usage count to prioritize unseen samples
            sorted_q = sorted(q, key=lambda i: self.sample_stats[i]['usage_count'])
            batch.extend(sorted_q[:samples_per_quantile])
        
        # Fill up the rest of the batch if batch_size is not divisible by 4
        while len(batch) < self.batch_size:
            remaining_indices = list(set(self.all_indices) - set(batch))
            if not remaining_indices: break
            batch.append(np.random.choice(remaining_indices))
            
        return batch

    def update_with_loss(self, batch_loss: float):
        """Update internal state after a training step."""
        if math.isnan(batch_loss) or batch_loss > 10 * (self.last_stable_loss + 1e-6):
            self._handle_gradient_explosion()
            return

        # Update stats for samples in the last batch
        for idx in self.current_batch_indices:
            stats = self.sample_stats[idx]
            stats['usage_count'] += 1
            # Use an exponential moving average for individual loss
            if stats['avg_loss'] == float('inf'):
                stats['avg_loss'] = batch_loss
            else:
                stats['avg_loss'] = 0.9 * stats['avg_loss'] + 0.1 * batch_loss
        
        self.last_stable_loss = (self.last_stable_loss * 9 + batch_loss) / 10
        self.global_step += 1
        self._update_lr()

        if self.ui:
            step_in_epoch = self.global_step % self.steps_per_epoch or self.steps_per_epoch
            self.ui.update(
                self.current_epoch + 1, self.num_epochs,
                step_in_epoch, self.steps_per_epoch,
                self.global_step, self.total_steps,
                batch_loss, self.lr
            )
    
    def _handle_gradient_explosion(self):
        """Recovers from a gradient explosion using CPU backup."""
        if self.ui:
            self.ui.log_message("Gradient explosion detected! Attempting recovery...")
        
        if self.cpu_backup and self.load_callback:
            self.load_callback(self.cpu_backup, from_ram=True)
            if self.ui:
                self.ui.log_message("Recovered model state from RAM backup.", color_pair=2)
            # Reduce LR temporarily
            self.lr *= 0.5 
        else:
            if self.ui:
                self.ui.log_message("No CPU backup found. Cannot recover automatically.")
    
    def update_cpu_backup(self, model_state):
        """User can call this to update the in-memory backup."""
        self.cpu_backup = model_state
        if self.ui:
            self.ui.log_message("CPU RAM backup updated.", color_pair=2)

    def epoch_iterator(self):
        """An iterator that yields for each epoch."""
        for i in range(self.num_epochs):
            self.current_epoch = i
            self.unseen_indices_in_epoch = list(self.all_indices)
            yield i

    def batch_iterator(self):
        """An iterator that yields batch indices and learning rate."""
        for _ in range(self.steps_per_epoch):
            if self.global_step >= self.total_steps:
                break
            
            self.current_batch_indices = self._get_next_batch_indices()
            yield self.current_batch_indices, self.lr
            
    def save_checkpoint(self, is_final=False):
        """Saves the scheduler and model state."""
        if not self.save_callback: return

        state = {
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'sample_stats': self.sample_stats,
            'initial_lr': self.initial_lr,
            'scheduler_mode': self.scheduler_mode
        }
        
        suffix = 'final' if is_final else f'epoch_{self.current_epoch}_step_{self.global_step}'
        scheduler_path = os.path.join(self.checkpoint_dir, f'scheduler_{suffix}.json')
        model_path = os.path.join(self.checkpoint_dir, f'model_{suffix}.bin')

        save_state_to_json(state, scheduler_path)
        self.save_callback(model_path) # Assumes callback gets filepath
        if self.ui:
            self.ui.log_message(f"Checkpoint saved to {self.checkpoint_dir}", color_pair=2)

    def load_checkpoint(self, filepath):
        """Loads the scheduler and model state from a checkpoint."""
        if not self.load_callback: return

        state = load_state_from_json(filepath)
        if state:
            # Restore scheduler state
            self.current_epoch = state.get('current_epoch', 0)
            self.global_step = state.get('global_step', 0)
            self.sample_stats = state.get('sample_stats', self.sample_stats)
            
            # Restore model state
            model_path = filepath.replace('scheduler_', 'model_').replace('.json', '.bin')
            self.load_callback(model_path)
            if self.ui:
                self.ui.log_message(f"Checkpoint loaded from {filepath}", color_pair=2)

    def close(self):
        """Cleans up resources, especially the UI."""
        if self.ui:
            self.ui.close()