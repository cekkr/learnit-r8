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
                 auto_find_lr: bool = False, # New feature flag
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
        
        # --- NEW: Dynamic LR Discovery ---
        self.auto_find_lr = auto_find_lr
        self.lr_is_finalized = not auto_find_lr # If not enabled, LR is considered final from the start
        self.optimal_lr_found = None if auto_find_lr else initial_lr

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
            self.num_epochs = 1000 # Virtual number for scheduling
        else:
            self.start_time = time.time()
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
        # If auto-finding, start with a much smaller LR
        self.lr = 1e-6 if self.auto_find_lr else self.initial_lr / 2.0
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

        config = {
            'num_samples': len(state['sample_stats']),
            'batch_size': state.get('batch_size', 32),
            'initial_lr': state.get('initial_lr', 0.01), # This will now be the found optimal LR
            'scheduler_mode': state.get('scheduler_mode', 'cosine'),
            'save_callback': save_callback,
            'load_callback': load_callback,
            'auto_find_lr': False, # Never auto-find on resume
            **kwargs
        }
        
        scheduler = cls(**config)
        
        # Restore state
        scheduler.current_epoch = state.get('current_epoch', 0)
        scheduler.global_step = state.get('global_step', 0)
        scheduler.sample_stats = state.get('sample_stats', {})
        scheduler.lr = state.get('lr', scheduler.initial_lr / 2.0)
        
        # Restore LR finder state
        scheduler.lr_is_finalized = state.get('lr_is_finalized', True)
        scheduler.optimal_lr_found = state.get('optimal_lr_found', scheduler.initial_lr)
        
        if scheduler.ui:
            scheduler.ui.log_message(f"Resumed training from {checkpoint_path}", color_pair=2)
            
        return scheduler

    def _update_lr(self):
        """Updates the learning rate, including the dynamic discovery phase."""
        # --- NEW: Dynamic LR Discovery Logic ---
        if self.auto_find_lr and not self.lr_is_finalized:
            # Aggressively increase LR until instability is detected
            # The actual finalization happens in _handle_gradient_explosion
            self.lr *= 1.05 # Exponential increase
            if self.ui:
                self.ui.log_message(f"LR Discovery Phase: Trying LR {self.lr:.8f}", color_pair=3)
            return

        # Standard warm-up and decay schedule (post-discovery)
        if self.global_step < self.warmup_steps:
            # Standard linear warm-up from half of the (now optimal) initial_lr
            start_lr = self.initial_lr / 2.0
            progress = self.global_step / self.warmup_steps
            self.lr = start_lr + (self.initial_lr - start_lr) * progress
        else:
            # Standard decay phase
            progress = (self.global_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            if self.scheduler_mode == 'cosine':
                self.lr = self.initial_lr * 0.5 * (1 + math.cos(math.pi * progress))
            else: # 'linear'
                self.lr = self.initial_lr * (1 - progress)
        
        self.lr = max(self.lr, 1e-8)
        
    def _handle_gradient_explosion(self):
        """Recovers from instability and finalizes LR if in discovery mode."""
        if self.ui: self.ui.log_message("Instability detected! Attempting recovery...", color_pair=4)
        
        exploded_lr = self.lr # The LR that caused the failure
        
        if self.cpu_backup and self.load_callback:
            self.load_callback(self.cpu_backup, from_ram=True)
            if self.ui: self.ui.log_message("Recovered model state from RAM backup.", color_pair=2)
            # Set LR to a safe value (half of the one that exploded)
            self.lr = exploded_lr / 2.0
            
            # --- NEW: Finalize LR Discovery ---
            if self.auto_find_lr and not self.lr_is_finalized:
                self.initial_lr = self.lr # This is our new peak LR
                self.optimal_lr_found = self.lr
                self.lr_is_finalized = True
                # Reset warm-up to start from this point
                self.warmup_steps = self.global_step + int(self.total_steps * self.warmup_proportion)
                if self.ui:
                    self.ui.log_message(f"Optimal LR Found: {self.initial_lr:.7f}. Finalizing schedule.", color_pair=2)

        else:
            if self.ui: self.ui.log_message("No CPU backup found. Cannot recover automatically.", color_pair=4)
            # If we can't recover, we can't continue, but at least report the finding
            if self.auto_find_lr and not self.lr_is_finalized:
                 raise RuntimeError(f"LR Finder failed due to gradient explosion at LR={exploded_lr:.7f} without a CPU backup to recover from.")


    def save_checkpoint(self, is_final=False):
        """Saves the scheduler and model state, including LR finder state."""
        if not self.save_callback: 
            return
        
        state = {
            'current_epoch': self.current_epoch, 
            'global_step': self.global_step, 
            'sample_stats': self.sample_stats, 
            'initial_lr': self.initial_lr, # Will now be the found optimal LR
            'scheduler_mode': self.scheduler_mode, 
            'batch_size': self.batch_size, 
            'lr': self.lr,
            # --- NEW: Save LR Finder State ---
            'lr_is_finalized': self.lr_is_finalized,
            'optimal_lr_found': self.optimal_lr_found
        }
        suffix = 'final' if is_final else f'epoch_{self.current_epoch}_step_{self.global_step}'
        scheduler_path = os.path.join(self.checkpoint_dir, f'scheduler_{suffix}.json')
        model_path = os.path.join(self.checkpoint_dir, f'model_{suffix}.bin')
        
        save_state_to_json(state, scheduler_path)
        self.save_callback(model_path)

        if self.ui:
            self.ui.log_message(f"Checkpoint saved to {self.checkpoint_dir}", color_pair=2)

    # --- Other methods like update_with_loss, _get_next_batch_indices, iterators, etc. remain the same ---
    # The existing update_cpu_backup is crucial for this feature to work.
    
    def update_with_loss(self, batch_loss: float):
        if math.isnan(batch_loss) or (self.last_stable_loss != float('inf') and batch_loss > 4 * self.last_stable_loss and self.global_step > 10):
            self._handle_gradient_explosion()
            return
            
        # Update CPU backup BEFORE the step that might explode
        if self.global_step % 10 == 0 and self.load_callback:
            # User must implement the logic to get the state dict
            # self.update_cpu_backup(model.state_dict())
            pass

        alpha = 0.1
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
            self.last_stable_loss = (self.last_stable_loss * 0.95 + batch_loss * 0.05)

        self.global_step += 1
        self._update_lr()

        if self.ui:
            step_in_epoch = (self.global_step - 1) % self.steps_per_epoch + 1
            total_display_steps = self.total_steps
            if self.train_duration_seconds:
                elapsed_time = time.time() - self.start_time
                if elapsed_time > 1:
                    steps_per_second = self.global_step / elapsed_time
                    total_display_steps = int(steps_per_second * self.train_duration_seconds)

            self.ui.update(
                epoch=self.current_epoch + 1, total_epochs=self.num_epochs,
                step=step_in_epoch, total_steps_in_epoch=self.steps_per_epoch,
                global_step=self.global_step, total_global_steps=total_display_steps,
                loss=batch_loss, lr=self.lr
            )
    
    def _get_next_batch_indices(self):
        """Intelligently selects indices for the next batch."""
        if self.current_epoch == 0 or not self.lr_is_finalized:
            available_indices = [i for i, count in self.epoch_usage_count.items() if count == 0]
            if not available_indices:
                self.epoch_usage_count = {i: 0 for i in range(self.num_samples)}
                available_indices = list(self.all_indices)
            batch = np.random.choice(available_indices, min(self.batch_size, len(available_indices)), replace=False).tolist()
        else:
            valid_samples = {i: s['avg_loss'] for i, s in self.sample_stats.items() if s['avg_loss'] != float('inf')}
            if not valid_samples:
                return np.random.choice(self.all_indices, self.batch_size, replace=False).tolist()
            target_avg_loss = np.mean(list(valid_samples.values()))
            sorted_indices = sorted(valid_samples.keys(), key=lambda i: valid_samples[i])
            quantiles = np.array_split(sorted_indices, 4)
            batch = []
            available_indices_in_epoch = [i for i, count in self.epoch_usage_count.items() if count < 2]
            for q in quantiles:
                candidates = [idx for idx in q if idx in available_indices_in_epoch]
                sorted_q = sorted(candidates, key=lambda i: self.epoch_usage_count[i])
                batch.extend(sorted_q[:self.batch_size // 4])
            if batch:
                batch = self._adjust_batch_to_target_loss(batch, target_avg_loss, available_indices_in_epoch)
        while len(batch) < self.batch_size:
            remaining_pool = [i for i in self.all_indices if i not in batch]
            if not remaining_pool: break
            batch.append(np.random.choice(remaining_pool))
        for idx in batch:
            self.epoch_usage_count[idx] += 1
        return batch

    def _adjust_batch_to_target_loss(self, batch, target_loss, available_pool, tolerance=0.1, max_swaps=5):
        batch_with_loss = [i for i in batch if self.sample_stats[i]['avg_loss'] != float('inf')]
        if not batch_with_loss: return batch
        current_loss = np.mean([self.sample_stats[i]['avg_loss'] for i in batch_with_loss])
        for _ in range(max_swaps):
            if target_loss > 0 and abs(current_loss - target_loss) / target_loss < tolerance: break
            error = current_loss - target_loss
            swap_pool = [i for i in available_pool if i not in batch and self.sample_stats[i]['avg_loss'] != float('inf')]
            if not swap_pool: break
            if error > 0:
                sample_to_remove = max(batch, key=lambda i: self.sample_stats[i].get('avg_loss', -1))
                sample_to_add = min(swap_pool, key=lambda i: self.sample_stats[i].get('avg_loss', float('inf')))
            else:
                sample_to_remove = min(batch, key=lambda i: self.sample_stats[i].get('avg_loss', float('inf')))
                sample_to_add = max(swap_pool, key=lambda i: self.sample_stats[i].get('avg_loss', -1))
            if sample_to_remove in batch:
                batch.remove(sample_to_remove)
                batch.append(sample_to_add)
                current_loss = np.mean([self.sample_stats[i]['avg_loss'] for i in batch if self.sample_stats[i]['avg_loss'] != float('inf')])
            else: break
        return batch
    
    def epoch_iterator(self):
        epoch_counter = self.current_epoch
        while True:
            if self.end_time and time.time() >= self.end_time: break
            if not self.end_time and epoch_counter >= self.num_epochs: break
            self.current_epoch = epoch_counter
            self.epoch_usage_count = {i: 0 for i in range(self.num_samples)}
            yield epoch_counter
            epoch_counter += 1

    def batch_iterator(self):
        max_steps = float('inf') if self.train_duration_seconds else self.steps_per_epoch
        step_counter = 0
        while step_counter < max_steps:
            if self.end_time and time.time() >= self.end_time: raise StopIteration
            if not self.train_duration_seconds and self.global_step >= self.total_steps: raise StopIteration
            self.current_batch_indices = self._get_next_batch_indices()
            yield self.current_batch_indices, self.lr
            step_counter += 1

    def load_checkpoint(self, filepath):
        if not self.load_callback: return
        state = load_state_from_json(filepath)
        if state:
            self.current_epoch = state.get('current_epoch', 0)
            self.global_step = state.get('global_step', 0)
            self.sample_stats = state.get('sample_stats', self.sample_stats)
            model_path = filepath.replace('scheduler_', 'model_').replace('.json', '.bin')
            self.load_callback(model_path)
            if self.ui: self.ui.log_message(f"Checkpoint loaded from {filepath}", color_pair=2)

    def update_cpu_backup(self, model_state):
        self.cpu_backup = copy.deepcopy(model_state)

    def close(self):
        if self.ui: self.ui.close()