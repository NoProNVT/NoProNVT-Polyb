import math
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Enhanced Cosine Annealing with Warm Restarts
def cosine_annealing_with_warmup_restarts(epoch, max_lr=1e-3, min_lr=1e-6, 
                                         warmup_epochs=5, cycle_epochs=50, restart_factor=0.8):
    """
    Cosine annealing with warm restarts and decay
    """
    if epoch < warmup_epochs:
        # Warmup phase
        return min_lr + (max_lr - min_lr) * epoch / warmup_epochs
    
    # Calculate cycle position
    cycle_epoch = (epoch - warmup_epochs) % cycle_epochs
    cycle_num = (epoch - warmup_epochs) // cycle_epochs
    
    # Apply restart decay
    current_max_lr = max_lr * (restart_factor ** cycle_num)
    
    # Cosine annealing within cycle
    cos_factor = 0.5 * (1 + math.cos(math.pi * cycle_epoch / cycle_epochs))
    lr = min_lr + (current_max_lr - min_lr) * cos_factor
    
    return lr

# One Cycle Learning Rate Schedule
def one_cycle_lr(epoch, max_lr=1e-3, total_epochs=100, warmup_pct=0.3, final_div=10):
    """
    One Cycle Learning Rate policy by Leslie Smith
    """
    warmup_epochs = int(total_epochs * warmup_pct)
    
    if epoch < warmup_epochs:
        # Warmup phase
        return max_lr * epoch / warmup_epochs
    elif epoch < total_epochs:
        # Annealing phase
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr = max_lr * (1 - progress) + (max_lr / final_div) * progress
        return lr
    else:
        return max_lr / final_div

# Exponential Decay with Steps
def exp_decay_steps(epoch, initial_lr=1e-3, decay_rate=0.9, decay_steps=10):
    """Exponential decay with steps"""
    return initial_lr * (decay_rate ** (epoch // decay_steps))

# Custom Learning Rate Scheduler Class
class CustomLRScheduler(tf.keras.callbacks.Callback):
    def __init__(self, schedule_func, verbose=1):
        super(CustomLRScheduler, self).__init__()
        self.schedule_func = schedule_func
        self.verbose = verbose
        
    def on_epoch_begin(self, epoch, logs=None):
        lr = self.schedule_func(epoch)
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        if self.verbose:
            print(f'Epoch {epoch+1}: Learning rate is {lr:.2e}')

# Enhanced Early Stopping with Learning Rate Reduction
class SmartEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', patience=15, min_delta=1e-4, 
                 restore_best_weights=True, lr_reduction_factor=0.5, 
                 lr_reduction_patience=7):
        super(SmartEarlyStopping, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.lr_reduction_factor = lr_reduction_factor
        self.lr_reduction_patience = lr_reduction_patience
        
        self.best_score = None
        self.wait = 0
        self.lr_wait = 0
        self.best_weights = None
        
    def on_epoch_end(self, epoch, logs=None):
        current_score = logs.get(self.monitor)
        
        if current_score is None:
            return
            
        # Check if we have improvement
        if self.best_score is None:
            self.best_score = current_score
            self.best_weights = self.model.get_weights()
            return
            
        # Determine if we improved
        improved = False
        if 'loss' in self.monitor:
            improved = current_score < self.best_score - self.min_delta
        else:
            improved = current_score > self.best_score + self.min_delta
            
        if improved:
            self.best_score = current_score
            self.wait = 0
            self.lr_wait = 0
            self.best_weights = self.model.get_weights()
            print(f"Epoch {epoch+1}: {self.monitor} improved to {current_score:.6f}")
        else:
            self.wait += 1
            self.lr_wait += 1
            
            # Reduce learning rate
            if self.lr_wait >= self.lr_reduction_patience:
                old_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
                new_lr = old_lr * self.lr_reduction_factor
                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                print(f"Epoch {epoch+1}: Reducing learning rate to {new_lr:.2e}")
                self.lr_wait = 0
                
            # Early stopping
            if self.wait >= self.patience:
                print(f"Epoch {epoch+1}: Early stopping triggered")
                if self.restore_best_weights and self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
                    print("Restored best weights")
                self.model.stop_training = True

# Model Checkpoint with Best Score Tracking
class EnhancedModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, monitor='val_loss', save_best_only=True, 
                 save_weights_only=True, save_freq='epoch'):
        super(EnhancedModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.save_freq = save_freq
        self.best_score = None
        
    def on_epoch_end(self, epoch, logs=None):
        current_score = logs.get(self.monitor)
        
        if current_score is None:
            return
            
        save_model = True
        if self.save_best_only:
            if self.best_score is None:
                self.best_score = current_score
            else:
                if 'loss' in self.monitor:
                    save_model = current_score < self.best_score
                else:
                    save_model = current_score > self.best_score
                    
                if save_model:
                    self.best_score = current_score
                    
        if save_model:
            if self.save_weights_only:
                self.model.save_weights(self.filepath, overwrite=True)
            else:
                self.model.save(self.filepath, overwrite=True)
            print(f"Epoch {epoch+1}: Model saved to {self.filepath}")

# Advanced Callbacks Setup
def get_callbacks(monitor='val_loss', mode='min', save_path='best_model.weights.h5',
                  max_lr=1e-3, min_lr=1e-6, total_epochs=100, 
                  schedule_type='one_cycle', save_weights_only=True,
                  early_stopping_patience=20, use_mixed_precision=False):
    """
    Get enhanced callbacks for optimal training
    """
    callbacks_list = []
    
    # Learning Rate Scheduler
    if schedule_type == 'one_cycle':
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: one_cycle_lr(epoch, max_lr, total_epochs), 
            verbose=1
        )
    elif schedule_type == 'cosine_restart':
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: cosine_annealing_with_warmup_restarts(epoch, max_lr, min_lr), 
            verbose=1
        )
    elif schedule_type == 'exp_decay':
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: exp_decay_steps(epoch, max_lr), 
            verbose=1
        )
    else:
        # Default cosine annealing
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: cosine_annealing_with_warmup_restarts(epoch, max_lr, min_lr), 
            verbose=1
        )
    
    callbacks_list.append(lr_scheduler)
    
    # Enhanced Early Stopping
    early_stopping = SmartEarlyStopping(
        monitor=monitor,
        patience=early_stopping_patience,
        restore_best_weights=True
    )
    callbacks_list.append(early_stopping)
    
    # Model Checkpoint
    checkpoint = EnhancedModelCheckpoint(
        filepath=save_path,
        monitor=monitor,
        save_best_only=True,
        save_weights_only=save_weights_only
    )
    callbacks_list.append(checkpoint)
    
    # CSV Logger for training history
    csv_logger = tf.keras.callbacks.CSVLogger(
        'training_history.csv', append=True
    )
    callbacks_list.append(csv_logger)
    
    # Reduce LR on Plateau (backup)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=0.5,
        patience=10,
        min_lr=min_lr,
        verbose=1,
        cooldown=5
    )
    callbacks_list.append(reduce_lr)
    
    # TensorBoard for visualization
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch'
    )
    callbacks_list.append(tensorboard)
    
    # Mixed Precision Loss Scaling (if enabled)
    if use_mixed_precision:
        try:
            from tensorflow.keras.mixed_precision import experimental as mixed_precision
            callbacks_list.append(mixed_precision.LossScaleOptimizer)
        except ImportError:
            print("Mixed precision not available")
    
    # NaN Termination
    nan_termination = tf.keras.callbacks.TerminateOnNaN()
    callbacks_list.append(nan_termination)
    
    return callbacks_list

# Plotting function for learning rate visualization
def plot_lr_schedule(schedule_func, epochs=100, save_path='lr_schedule.png'):
    """Plot learning rate schedule"""
    epochs_range = range(epochs)
    lrs = [schedule_func(epoch) for epoch in epochs_range]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, lrs, 'b-', linewidth=2)
    plt.title('Learning Rate Schedule', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Learning Rate', fontsize=14)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Learning rate schedule saved to {save_path}")

# Kaggle-specific callback for memory management
class MemoryCallback(tf.keras.callbacks.Callback):
    """Monitor memory usage during training"""
    
    def on_epoch_end(self, epoch, logs=None):
        import gc
        import psutil
        import os
        
        # Force garbage collection
        gc.collect()
        
        # Get memory usage
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Epoch {epoch+1}: Memory usage: {memory_usage:.1f} MB")
        
        # Clear TensorFlow cache if memory usage is high
        if memory_usage > 8000:  # 8GB threshold
            tf.keras.backend.clear_session()
            print("Cleared TensorFlow session due to high memory usage")

def get_kaggle_callbacks(monitor='val_loss', save_path='best_model.weights.h5',
                        max_lr=2e-3, total_epochs=100):
    """
    Optimized callbacks specifically for Kaggle environment
    """
    return get_callbacks(
        monitor=monitor,
        save_path=save_path,
        max_lr=max_lr,
        min_lr=1e-6,
        total_epochs=total_epochs,
        schedule_type='one_cycle',
        early_stopping_patience=25,
        use_mixed_precision=True
    ) + [MemoryCallback()]
