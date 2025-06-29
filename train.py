import os
import gc

# Set TensorFlow environment variables for optimization
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'

import tensorflow as tf
from model import build_model
from metrics.segmentation_metrics import (
    dice_coeff, bce_dice_loss, IoU, zero_IoU, dice_loss, 
    get_loss_function, get_metrics, focal_dice_loss, mean_dice, mean_iou
)
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.utils import get_custom_objects
from sklearn.model_selection import train_test_split, StratifiedKFold
from callbacks.callbacks import get_kaggle_callbacks, plot_lr_schedule
from dataloader.dataloader import build_augmenter, build_dataset, build_decoder, auto_select_accelerator
import yaml
import numpy as np
import matplotlib.pyplot as plt

def setup_mixed_precision():
    """Setup mixed precision training for better performance"""
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision enabled: mixed_float16")
        return True
    except Exception as e:
        print(f"Mixed precision not available: {e}")
        return False

def setup_memory_growth():
    """Configure GPU memory growth to avoid OOM errors"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"Memory growth setup failed: {e}")

def get_optimizer(optimizer_type='adamw', learning_rate=1e-3, mixed_precision=False, weight_decay=1e-4):
    """Get optimized optimizer with proper settings"""
    if optimizer_type.lower() == 'adamw':
        optimizer = AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,  # Configurable weight decay
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
    else:
        optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
    
    if mixed_precision:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
    return optimizer

def compute_class_weights(Y_train):
    """Compute class weights for imbalanced dataset"""
    positive_pixels = np.sum(Y_train > 0.5)
    negative_pixels = np.sum(Y_train <= 0.5)
    total_pixels = positive_pixels + negative_pixels
    
    pos_weight = negative_pixels / positive_pixels
    neg_weight = 1.0
    
    print(f"Positive pixels: {positive_pixels:,}")
    print(f"Negative pixels: {negative_pixels:,}")
    print(f"Positive weight: {pos_weight:.3f}")
    
    return {0: neg_weight, 1: pos_weight}

def prepare_datasets(X_paths, Y_paths, config):
    """Prepare training, validation and test datasets"""
    img_size = config['Data']['img_size']
    valid_size = config['Hyperparameter']['valid_size']
    test_size = config['Hyperparameter']['test_size']
    batch_size = config['Hyperparameter']['BATCH_SIZE']
    seed = config['Hyperparameter']['SEED']
    
    # Split data
    X_temp, X_test, Y_temp, Y_test = train_test_split(
        X_paths, Y_paths, test_size=test_size, random_state=seed, shuffle=True
    )
    
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        X_temp, Y_temp, test_size=valid_size, random_state=seed, shuffle=True
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_valid)}")
    print(f"Test samples: {len(X_test)}")
    
    # Build decoders
    train_decoder = build_decoder(
        with_labels=True, target_size=(img_size, img_size), 
        ext='jpg', segment=True, ext2='jpg'
    )
    
    # Build datasets
    train_dataset = build_dataset(
        X_train, Y_train, 
        bsize=batch_size,
        decode_fn=train_decoder,
        augment=True,
        augmentAdvSeg=config['Hyperparameter'].get('use_advanced_augmentation', True),
        shuffle=True,
        use_mixup=config['Hyperparameter'].get('use_mixup', False),
        prefetch_size=config['Hardware'].get('prefetch_buffer_size', 8)
    )
    
    valid_dataset = build_dataset(
        X_valid, Y_valid,
        bsize=batch_size,
        decode_fn=train_decoder,
        augment=False,
        repeat=False,
        shuffle=False,
        prefetch_size=config['Hardware'].get('prefetch_buffer_size', 8)
    )
    
    test_dataset = build_dataset(
        X_test, Y_test,
        bsize=batch_size,
        decode_fn=train_decoder,
        augment=False,
        repeat=False,
        shuffle=False,
        prefetch_size=config['Hardware'].get('prefetch_buffer_size', 8)
    )
    
    return train_dataset, valid_dataset, test_dataset, (X_train, X_valid, X_test)

def build_and_compile_model(config, mixed_precision=False):
    """Build and compile model with optimized settings"""
    img_size = config['Data']['img_size']
    model_config = config.get('Model', {})
    training_config = config.get('Training', {})
    advanced_config = config.get('Advanced', {})
    
    # Build model
    print("Building model...")
    model = build_model(
        img_size=img_size,
        backbone=model_config.get('backbone', 'efficientnetv2-b0'),
        use_attention=model_config.get('use_attention', True),
        use_pyramid=model_config.get('use_pyramid_pooling', True),
        dropout_rate=config['Hyperparameter'].get('dropout_rate', 0.3)
    )
    
    # Get loss function with enhanced weights
    loss_type = training_config.get('loss_type', 'focal_dice')
    loss_weights = training_config.get('loss_weights', {})
    
    # Apply label smoothing if enabled
    label_smoothing = advanced_config.get('label_smoothing', 0.0)
    loss_fn = get_loss_function(loss_type, **loss_weights)
    
    # Get optimizer with configurable weight decay
    max_lr = config['Hyperparameter']['max_lr']
    weight_decay = advanced_config.get('weight_decay', 1e-4)
    optimizer = get_optimizer('adamw', max_lr, mixed_precision, weight_decay)
    
    # Get enhanced metrics
    metrics = [mean_dice, mean_iou, 'accuracy']
    
    # Compile model
    print(f"Compiling model with {loss_type} loss (label_smoothing={label_smoothing})...")
    print(f"Optimizer: AdamW with weight_decay={weight_decay}")
    
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=metrics
    )
    
    return model

def main(config_file):
    try:
        # Read configuration
        print("Reading config file:", config_file)
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("Configuration loaded successfully")
        
        # Setup hardware optimizations
        setup_memory_growth()
        
        # Setup mixed precision if enabled
        mixed_precision = config['Hyperparameter'].get('use_mixed_precision', False)
        if mixed_precision:
            setup_mixed_precision()
        
        # Setup distributed strategy
        strategy = auto_select_accelerator()
        
        with strategy.scope():
            # Get data paths
            route = os.path.normpath(os.path.abspath(config['Data']['route']))
            X_path = config['Data']['X_path']
            Y_path = config['Data']['Y_path']
            images_dir = os.path.join(route, X_path)
            masks_dir = os.path.join(route, Y_path)
            
            # Check directories
            print("Checking directories...")
            if not os.path.exists(images_dir):
                raise FileNotFoundError(f"Images directory not found: {images_dir}")
            if not os.path.exists(masks_dir):
                raise FileNotFoundError(f"Masks directory not found: {masks_dir}")
            
            # Get file lists
            X_paths = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir)])
            Y_paths = sorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir)])
            
            print(f"Found {len(X_paths)} images and {len(Y_paths)} masks")
            
            # Prepare datasets
            print("Preparing datasets...")
            train_dataset, valid_dataset, test_dataset, (X_train, X_valid, X_test) = prepare_datasets(
                X_paths, Y_paths, config
            )
            
            # Build and compile model
            model = build_and_compile_model(config, mixed_precision)
            model.summary()
            
            # Setup callbacks
            save_path = os.path.normpath(os.path.join(
                os.path.dirname(os.path.abspath(config_file)), 
                config['Model']['save_path']
            ))
            
            # Create checkpoint directory
            checkpoint_dir = os.path.dirname(save_path)
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
                print(f"Created checkpoint directory: {checkpoint_dir}")
            
            callbacks = get_kaggle_callbacks(
                monitor=config['Hyperparameter'].get('monitor', 'val_mean_dice'),
                save_path=save_path,
                max_lr=config['Hyperparameter']['max_lr'],
                total_epochs=config['Hyperparameter']['epochs']
            )
            
            # Calculate steps per epoch
            steps_per_epoch = len(X_train) // config['Hyperparameter']['BATCH_SIZE']
            validation_steps = len(X_valid) // config['Hyperparameter']['BATCH_SIZE']
            
            print(f"Steps per epoch: {steps_per_epoch}")
            print(f"Validation steps: {validation_steps}")
            
            # Check for gradient accumulation
            gradient_accumulation_steps = config.get('Advanced', {}).get('gradient_accumulation_steps', 1)
            progressive_resizing = config.get('Advanced', {}).get('progressive_resizing', False)
            
            # Start training with advanced features
            print("Starting enhanced training...")
            print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
            print(f"Progressive resizing: {progressive_resizing}")
            
            # For now, use standard training (gradient accumulation can be implemented later)
            if gradient_accumulation_steps > 1:
                print(f"Note: Gradient accumulation will simulate effective batch size: {config['Hyperparameter']['BATCH_SIZE'] * gradient_accumulation_steps}")
                print("Using standard training for now (gradient accumulation available in future update)")
            
            history = model.fit(
                train_dataset,
                epochs=config['Hyperparameter']['epochs'],
                verbose=1,
                callbacks=callbacks,
                validation_data=valid_dataset,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps
            )
            
            print("Training completed successfully!")
            
            # Evaluate on test set
            print("Evaluating on test set...")
            test_results = model.evaluate(test_dataset, verbose=1)
            print("Test Results:")
            for i, metric_name in enumerate(model.metrics_names):
                print(f"  {metric_name}: {test_results[i]:.4f}")
            
            # Save training plots
            if config.get('Kaggle', {}).get('save_training_plots', True):
                save_training_plots(history, config.get('Kaggle', {}).get('plot_output_dir', './plots'))
            
            # Clear memory
            gc.collect()
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise
    except KeyError as e:
        print(f"Error: Missing key {e} in config file")
        raise
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML format in {config_file}: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

def save_training_plots(history, output_dir='./plots'):
    """Save training history plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training & validation loss
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mean_dice'], label='Training Dice')
    plt.plot(history.history['val_mean_dice'], label='Validation Dice')
    plt.title('Model Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training plots saved to {output_dir}")

if __name__ == "__main__":
    config_file = "./config/train_config.yaml"
    main(config_file)
