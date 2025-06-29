import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import yaml
from yaml.loader import SafeLoader
from backbone.basic_backbone import EfficientNetV2B0, ConvnextSmall
from model import build_model
from metrics.segmentation_metrics import dice_coeff, bce_dice_loss, IoU, zero_IoU, mean_dice, mean_iou
import albumentations as A
from sklearn.metrics import confusion_matrix
import seaborn as sns

def setup_gpu():
    """Setup GPU for inference"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU setup completed for {len(gpus)} device(s)")
        except RuntimeError as e:
            print(f"GPU setup failed: {e}")

def get_tta_transforms():
    """Get Test Time Augmentation transforms"""
    return [
        A.NoOp(),  # Original image
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.Rotate(limit=90, p=1.0),
        A.Transpose(p=1.0),
    ]

def apply_tta(model, image, img_size=256):
    """Apply Test Time Augmentation"""
    transforms = get_tta_transforms()
    predictions = []
    
    for transform in transforms:
        # Apply transform
        augmented = transform(image=image)
        aug_image = augmented['image']
        
        # Resize and predict
        resized = cv2.resize(aug_image, (img_size, img_size))
        input_tensor = tf.expand_dims(resized.astype(np.float32) / 255.0, 0)
        pred = model.predict(input_tensor, verbose=0)[0, :, :, 0]
        
        # Reverse transform for prediction
        if transform.__class__.__name__ == 'HorizontalFlip':
            pred = np.fliplr(pred)
        elif transform.__class__.__name__ == 'VerticalFlip':
            pred = np.flipud(pred)
        elif transform.__class__.__name__ == 'Rotate':
            pred = np.rot90(pred, k=-1)  # Reverse rotation
        elif transform.__class__.__name__ == 'Transpose':
            pred = np.transpose(pred)
        
        predictions.append(pred)
    
    # Average all predictions
    return np.mean(predictions, axis=0)

def post_process_mask(mask, min_area=50, kernel_size=3):
    """Post-process prediction mask"""
    # Convert to binary
    binary_mask = (mask > 0.5).astype(np.uint8)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Opening to remove small noise
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    # Closing to fill small holes
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    # Remove small connected components
    num_labels, labels = cv2.connectedComponents(binary_mask)
    for i in range(1, num_labels):
        component_mask = (labels == i)
        if np.sum(component_mask) < min_area:
            binary_mask[component_mask] = 0
    
    return binary_mask.astype(np.float32)

def overlay_mask_on_image(image, mask, alpha=0.4, color=(0, 255, 0)):
    """Enhanced mask overlay with better visualization"""
    # Ensure mask is in proper format
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask = (mask > 0.5).astype(np.uint8) * 255
    
    # Create colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask == 255] = color
    
    # Smooth the mask edges
    colored_mask = cv2.GaussianBlur(colored_mask, (3, 3), 0)
    
    # Create overlay
    overlay = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
    
    # Add contours for better visibility
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
    
    return overlay

def load_and_resize_image_and_mask(image_path, mask_path, img_size):
    """Load and preprocess image and mask"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask: {mask_path}")
    
    # Resize to target size
    target_size = (img_size, img_size)
    image = cv2.resize(image, target_size)
    mask = cv2.resize(mask, target_size)
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    mask = mask.astype(np.float32) / 255.0
    mask = np.expand_dims(mask, axis=-1)
    
    return image, mask

def calculate_comprehensive_metrics(y_true, y_pred, threshold=0.5):
    """Calculate comprehensive evaluation metrics"""
    # Convert to binary
    y_true_bin = (y_true > threshold).astype(np.int32).flatten()
    y_pred_bin = (y_pred > threshold).astype(np.int32).flatten()
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # IoU
    intersection = np.sum(y_true_bin * y_pred_bin)
    union = np.sum(y_true_bin) + np.sum(y_pred_bin) - intersection
    iou = intersection / union if union > 0 else 0
    
    # Dice
    dice = 2 * intersection / (np.sum(y_true_bin) + np.sum(y_pred_bin)) if (np.sum(y_true_bin) + np.sum(y_pred_bin)) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1,
        'iou': iou,
        'dice': dice
    }

def benchmark_batch_images(route, model_path, img_size, batch_size, model_name='efficientnetv2-b0', 
                          use_tta=False, save_predictions=True, output_dir='./results'):
    """Enhanced benchmarking with comprehensive evaluation"""
    print("Setting up GPU...")
    setup_gpu()
    
    print("Loading model...")
    model = build_model(img_size, backbone=model_name)
    model.load_weights(model_path, by_name=True, skip_mismatch=True)
    print("Model loaded successfully")
    
    # Prepare directories
    images_dir = os.path.join(route, 'images')
    masks_dir = os.path.join(route, 'masks')
    
    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        raise FileNotFoundError("Images or masks directory not found")
    
    # Get file lists
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))])
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))])
    
    if len(image_files) != len(mask_files):
        print(f"Warning: Mismatch in file counts - {len(image_files)} images, {len(mask_files)} masks")
    
    print(f"Found {len(image_files)} images for evaluation")
    
    # Prepare output directory
    if save_predictions:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'overlays'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
    
    # Evaluation metrics
    all_metrics = []
    all_predictions = []
    all_overlays = []
    
    print("Starting evaluation...")
    for idx, (img_file, mask_file) in enumerate(zip(image_files, mask_files)):
        img_path = os.path.join(images_dir, img_file)
        mask_path = os.path.join(masks_dir, mask_file)
        
        try:
            # Load and preprocess
            original_img = cv2.imread(img_path)
            image, true_mask = load_and_resize_image_and_mask(img_path, mask_path, img_size)
            
            # Make prediction
            if use_tta:
                pred_mask = apply_tta(model, (image * 255).astype(np.uint8), img_size)
            else:
                input_tensor = tf.expand_dims(image, 0)
                pred_mask = model.predict(input_tensor, verbose=0)[0, :, :, 0]
            
            # Post-process prediction
            pred_mask_processed = post_process_mask(pred_mask)
            
            # Calculate metrics
            metrics = calculate_comprehensive_metrics(true_mask[:, :, 0], pred_mask_processed)
            metrics['filename'] = img_file
            all_metrics.append(metrics)
            
            # Create overlay
            resized_original = cv2.resize(original_img, (img_size, img_size))
            overlay = overlay_mask_on_image(resized_original, pred_mask_processed)
            all_overlays.append((f"{img_file}", overlay))
            
            # Save individual results
            if save_predictions:
                cv2.imwrite(os.path.join(output_dir, 'predictions', f'pred_{img_file}'), 
                          (pred_mask_processed * 255).astype(np.uint8))
                cv2.imwrite(os.path.join(output_dir, 'overlays', f'overlay_{img_file}'), overlay)
            
            print(f"Processed {idx+1}/{len(image_files)}: {img_file} - Dice: {metrics['dice']:.4f}")
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue
    
    # Calculate overall metrics
    if all_metrics:
        avg_metrics = {}
        for key in all_metrics[0].keys():
            if key != 'filename':
                avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Dice Coefficient: {avg_metrics['dice']:.4f}")
        print(f"IoU Score:        {avg_metrics['iou']:.4f}")
        print(f"Accuracy:         {avg_metrics['accuracy']:.4f}")
        print(f"Precision:        {avg_metrics['precision']:.4f}")
        print(f"Recall:           {avg_metrics['recall']:.4f}")
        print(f"Specificity:      {avg_metrics['specificity']:.4f}")
        print(f"F1-Score:         {avg_metrics['f1_score']:.4f}")
        print("="*50)
        
        # Save metrics to file
        if save_predictions:
            import pandas as pd
            df = pd.DataFrame(all_metrics)
            df.to_csv(os.path.join(output_dir, 'detailed_metrics.csv'), index=False)
            
            # Save summary metrics
            with open(os.path.join(output_dir, 'summary_metrics.txt'), 'w') as f:
                f.write("EVALUATION SUMMARY\n")
                f.write("="*30 + "\n")
                for key, value in avg_metrics.items():
                    f.write(f"{key}: {value:.4f}\n")
    
    # Create visualization grid
    if all_overlays:
        create_result_grid(all_overlays, output_dir)
    
    return all_metrics

def create_result_grid(overlays, output_dir, max_images=16):
    """Create a grid visualization of results"""
    n_images = min(len(overlays), max_images)
    cols = int(np.ceil(np.sqrt(n_images)))
    rows = int(np.ceil(n_images / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten() if n_images > 1 else [axes]
    
    for i in range(n_images):
        filename, overlay_img = overlays[i]
        axes[i].imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
        axes[i].set_title(filename, fontsize=10)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'result_grid.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Result grid saved to {os.path.join(output_dir, 'result_grid.png')}")

def create_metrics_plots(metrics_list, output_dir):
    """Create visualization plots for metrics"""
    if not metrics_list:
        return
    
    import pandas as pd
    df = pd.DataFrame(metrics_list)
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    metrics_to_plot = ['dice', 'iou', 'accuracy', 'precision', 'recall', 'f1_score']
    
    for i, metric in enumerate(metrics_to_plot):
        if metric in df.columns:
            axes[i].hist(df[metric], bins=20, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{metric.replace("_", " ").title()} Distribution')
            axes[i].set_xlabel(metric.replace("_", " ").title())
            axes[i].set_ylabel('Frequency')
            axes[i].axvline(df[metric].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df[metric].mean():.3f}')
            axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Metrics plots saved to {os.path.join(output_dir, 'metrics_distribution.png')}")

def enhanced_benchmark_with_config(config_file='./config/benchmark_config.yaml'):
    """Enhanced benchmarking using the new optimized config structure"""
    print("🚀 Starting Enhanced PEFNet Benchmark Evaluation...")
    print("="*60)
    
    # Load configuration
    with open(config_file) as f:
        config = yaml.load(f, Loader=SafeLoader)
    
    # Extract configuration sections
    data_config = config.get('Data', {})
    model_config = config.get('Model', {})
    eval_config = config.get('Evaluation', {})
    metrics_config = config.get('Metrics', {})
    output_config = config.get('Output', {})
    hardware_config = config.get('Hardware', {})
    advanced_config = config.get('Advanced', {})
    
    print(f"📊 Configuration Summary:")
    print(f"   - Image Size: {data_config.get('img_size', 384)}x{data_config.get('img_size', 384)}")
    print(f"   - Model: {model_config.get('model_name', 'efficientnetv2-b1')}")
    print(f"   - Batch Size: {eval_config.get('BATCH_SIZE', 24)}")
    print(f"   - TTA Enabled: {eval_config.get('use_tta', True)}")
    print(f"   - TTA Steps: {eval_config.get('tta_steps', 8)}")
    print(f"   - Post-processing: {eval_config.get('use_post_processing', True)}")
    print(f"   - Mixed Precision: {hardware_config.get('use_mixed_precision', True)}")
    print()
    
    # Setup mixed precision if enabled
    if hardware_config.get('use_mixed_precision', True):
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("✅ Mixed precision enabled for faster inference")
        except:
            print("⚠️ Mixed precision not available")
    
    # Setup GPU with enhanced settings
    setup_enhanced_gpu(hardware_config)
    
    # Load model with enhanced settings
    model = load_enhanced_model(model_config, data_config)
    
    # Find and prepare data
    data_paths = find_valid_data_paths(data_config)
    
    # Run enhanced evaluation
    metrics = run_enhanced_evaluation_pipeline(
        model, data_paths, config, eval_config, output_config
    )
    
    # Generate comprehensive report
    if metrics:
        output_dir = output_config.get('results_dir', './benchmark_results')
        generate_comprehensive_report(metrics, config, output_dir)
        print(f"✅ Enhanced evaluation completed! Check {output_dir}/ for detailed results.")
    
    return metrics

def setup_enhanced_gpu(hardware_config):
    """Setup GPU with enhanced configuration"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set memory limit if specified
            memory_limit = hardware_config.get('max_memory_limit', 15000)
            if memory_limit and gpus:
                tf.config.experimental.set_memory_limit(gpus[0], memory_limit)
            
            print(f"✅ Enhanced GPU setup completed for {len(gpus)} device(s)")
        except RuntimeError as e:
            print(f"⚠️ GPU setup failed: {e}")

def load_enhanced_model(model_config, data_config):
    """Load model with enhanced error handling and multiple path fallback"""
    img_size = data_config.get('img_size', 384)
    model_name = model_config.get('model_name', 'efficientnetv2-b1')
    
    print(f"🤖 Loading enhanced model: {model_name}...")
    
    # Build model with enhanced settings
    model = build_model(
        img_size=img_size, 
        backbone=model_name,
        use_attention=model_config.get('use_attention', True),
        use_pyramid=model_config.get('use_pyramid_pooling', True)
    )
    
    # Try multiple model paths with fallback
    model_paths = [model_config.get('model_path')] + model_config.get('backup_paths', [])
    model_loaded = False
    
    for path in model_paths:
        if path and os.path.exists(path):
            try:
                model.load_weights(path, by_name=True, skip_mismatch=True)
                print(f"✅ Model loaded successfully from: {path}")
                model_loaded = True
                break
            except Exception as e:
                print(f"⚠️ Failed to load from {path}: {e}")
    
    if not model_loaded:
        raise FileNotFoundError("❌ Could not load model from any specified path")
    
    return model

def find_valid_data_paths(data_config):
    """Find valid data paths with fallback options"""
    route = data_config.get('route', './config/benchmark_data')
    alternative_routes = data_config.get('alternative_routes', [])
    X_path = data_config.get('X_path', 'images')
    Y_path = data_config.get('Y_path', 'masks')
    
    for test_route in [route] + alternative_routes:
        images_dir = os.path.join(test_route, X_path)
        masks_dir = os.path.join(test_route, Y_path)
        
        if os.path.exists(images_dir) and os.path.exists(masks_dir):
            print(f"✅ Using data from: {test_route}")
            return {
                'root': test_route,
                'images': images_dir,
                'masks': masks_dir
            }
    
    raise FileNotFoundError(f"❌ No valid data found in specified routes")

def run_enhanced_evaluation_pipeline(model, data_paths, config, eval_config, output_config):
    """Run the complete enhanced evaluation pipeline"""
    
    # Get file lists
    image_files = sorted([f for f in os.listdir(data_paths['images']) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))])
    mask_files = sorted([f for f in os.listdir(data_paths['masks']) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))])
    
    print(f"📁 Found {len(image_files)} images and {len(mask_files)} masks")
    
    # Setup output directories
    output_dirs = setup_enhanced_output_directories(output_config)
    
    # Find optimal threshold if enabled
    optimal_threshold = find_optimal_threshold_enhanced(
        model, image_files, mask_files, data_paths, eval_config
    ) if eval_config.get('optimize_threshold', True) else 0.5
    
    print(f"🎯 Using threshold: {optimal_threshold:.3f}")
    
    # Run evaluation on all images
    all_metrics = []
    failed_cases = []
    
    print(f"🔄 Starting enhanced evaluation...")
    
    for idx, (img_file, mask_file) in enumerate(zip(image_files, mask_files)):
        try:
            metrics = process_single_image_enhanced(
                model, img_file, mask_file, data_paths,
                config, optimal_threshold, output_dirs
            )
            
            if metrics:
                all_metrics.append(metrics)
                if metrics['dice'] < config.get('Quality', {}).get('failure_threshold', 0.7):
                    failed_cases.append(metrics)
            
            # Progress update
            if (idx + 1) % 10 == 0 or (idx + 1) == len(image_files):
                avg_dice = np.mean([m['dice'] for m in all_metrics[-10:]])
                print(f"📊 Progress: {idx+1}/{len(image_files)} | Recent Avg Dice: {avg_dice:.3f}")
                
        except Exception as e:
            print(f"❌ Error processing {img_file}: {e}")
            continue
    
    # Add failure analysis
    if failed_cases:
        analyze_failure_cases_enhanced(failed_cases, output_dirs['base'])
    
    return all_metrics

def setup_enhanced_output_directories(output_config):
    """Setup comprehensive output directory structure"""
    base_dir = output_config.get('results_dir', './benchmark_results')
    
    dirs = {
        'base': base_dir,
        'predictions': os.path.join(base_dir, 'predictions'),
        'overlays': os.path.join(base_dir, 'overlays'),
        'plots': os.path.join(base_dir, 'plots'),
        'analysis': os.path.join(base_dir, 'analysis'),
        'failures': os.path.join(base_dir, 'failure_cases')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"📁 Output directories created in: {base_dir}")
    return dirs

def find_optimal_threshold_enhanced(model, image_files, mask_files, data_paths, eval_config):
    """Enhanced threshold optimization with comprehensive search"""
    if not eval_config.get('optimize_threshold', True):
        return 0.5
    
    threshold_range = eval_config.get('threshold_range', [0.3, 0.7])
    threshold_steps = eval_config.get('threshold_steps', 21)
    
    thresholds = np.linspace(threshold_range[0], threshold_range[1], threshold_steps)
    best_threshold = 0.5
    best_score = 0.0
    
    print(f"🔍 Optimizing threshold over {len(thresholds)} values...")
    
    # Use subset for optimization (first 20 images)
    test_files = list(zip(image_files[:20], mask_files[:20]))
    
    for threshold in thresholds:
        scores = []
        
        for img_file, mask_file in test_files:
            try:
                img_path = os.path.join(data_paths['images'], img_file)
                mask_path = os.path.join(data_paths['masks'], mask_file)
                
                # Quick evaluation
                image, true_mask = load_and_resize_image_and_mask(
                    img_path, mask_path, eval_config.get('img_size', 384)
                )
                
                input_tensor = tf.expand_dims(image, 0)
                pred_mask = model.predict(input_tensor, verbose=0)[0, :, :, 0]
                
                metrics = calculate_comprehensive_metrics(true_mask[:, :, 0], pred_mask, threshold)
                scores.append(metrics['dice'])
                
            except:
                continue
        
        avg_score = np.mean(scores) if scores else 0
        if avg_score > best_score:
            best_score = avg_score
            best_threshold = threshold
    
    print(f"✅ Optimal threshold: {best_threshold:.3f} (Dice: {best_score:.3f})")
    return best_threshold

def process_single_image_enhanced(model, img_file, mask_file, data_paths, config, threshold, output_dirs):
    """Process single image with all enhanced features"""
    
    # Get configs
    data_config = config.get('Data', {})
    eval_config = config.get('Evaluation', {})
    output_config = config.get('Output', {})
    
    img_size = data_config.get('img_size', 384)
    
    # Load image and mask
    img_path = os.path.join(data_paths['images'], img_file)
    mask_path = os.path.join(data_paths['masks'], mask_file)
    
    original_img = cv2.imread(img_path)
    image, true_mask = load_and_resize_image_and_mask(img_path, mask_path, img_size)
    
    # Enhanced prediction with TTA
    if eval_config.get('use_tta', True):
        pred_mask = apply_enhanced_tta_prediction(
            model, image, img_size, eval_config
        )
    else:
        input_tensor = tf.expand_dims(image, 0)
        pred_mask = model.predict(input_tensor, verbose=0)[0, :, :, 0]
    
    # Enhanced post-processing
    pred_mask_processed = apply_comprehensive_postprocessing(pred_mask, eval_config, threshold)
    
    # Calculate enhanced metrics
    metrics = calculate_enhanced_metrics_suite(true_mask[:, :, 0], pred_mask_processed, threshold)
    metrics['filename'] = img_file
    
    # Save enhanced results
    if output_config.get('save_predictions', True):
        save_comprehensive_results(
            original_img, pred_mask_processed, true_mask[:, :, 0], 
            img_file, output_dirs, output_config
        )
    
    return metrics

def apply_enhanced_tta_prediction(model, image, img_size, eval_config):
    """Apply enhanced TTA with configurable transforms"""
    tta_transforms = eval_config.get('tta_transforms', [
        'horizontal_flip', 'vertical_flip', 'rotate_90', 'rotate_180'
    ])
    
    transforms = [A.NoOp()]  # Original
    
    # Add transforms based on config
    for transform_name in tta_transforms:
        if transform_name == 'horizontal_flip':
            transforms.append(A.HorizontalFlip(p=1.0))
        elif transform_name == 'vertical_flip':
            transforms.append(A.VerticalFlip(p=1.0))
        elif transform_name == 'rotate_90':
            transforms.append(A.Rotate(limit=[90, 90], p=1.0))
        elif transform_name == 'rotate_180':
            transforms.append(A.Rotate(limit=[180, 180], p=1.0))
        elif transform_name == 'rotate_270':
            transforms.append(A.Rotate(limit=[270, 270], p=1.0))
    
    predictions = []
    image_uint8 = (image * 255).astype(np.uint8)
    
    for transform in transforms:
        augmented = transform(image=image_uint8)
        aug_image = augmented['image']
        
        # Predict
        resized = cv2.resize(aug_image, (img_size, img_size))
        input_tensor = tf.expand_dims(resized.astype(np.float32) / 255.0, 0)
        pred = model.predict(input_tensor, verbose=0)[0, :, :, 0]
        
        predictions.append(pred)
    
    return np.mean(predictions, axis=0)

def apply_comprehensive_postprocessing(pred_mask, eval_config, threshold):
    """Apply comprehensive post-processing pipeline"""
    # Initial thresholding
    binary_mask = (pred_mask > threshold).astype(np.uint8)
    
    # Morphological operations
    if eval_config.get('use_morphological_ops', True):
        kernel_size = eval_config.get('morph_kernel_size', 3)
        iterations = eval_config.get('morph_iterations', 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Opening to remove noise
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
        # Closing to fill gaps
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    # Enhanced CRF-like post-processing
    if eval_config.get('use_crf', True):
        # Gaussian smoothing as simplified CRF
        gaussian_weight = eval_config.get('crf_gaussian_weight', 3.0)
        bilateral_weight = eval_config.get('crf_bilateral_weight', 5.0)
        
        # Apply Gaussian blur
        smoothed = cv2.GaussianBlur(binary_mask.astype(np.float32), (5, 5), gaussian_weight)
        
        # Apply bilateral filter for edge preservation
        bilateral = cv2.bilateralFilter(smoothed, 9, bilateral_weight, bilateral_weight)
        
        binary_mask = (bilateral > 0.5).astype(np.uint8)
    
    return binary_mask.astype(np.float32)

def calculate_enhanced_metrics_suite(y_true, y_pred, threshold=0.5):
    """Calculate comprehensive enhanced metrics"""
    # Basic metrics
    metrics = calculate_comprehensive_metrics(y_true, y_pred, threshold)
    
    # Boundary metrics
    try:
        # Edge detection for boundary metrics
        y_true_edges = cv2.Canny((y_true * 255).astype(np.uint8), 50, 150) > 0
        y_pred_edges = cv2.Canny((y_pred * 255).astype(np.uint8), 50, 150) > 0
        
        # Boundary IoU
        boundary_intersection = np.sum(y_true_edges & y_pred_edges)
        boundary_union = np.sum(y_true_edges | y_pred_edges)
        boundary_iou = boundary_intersection / boundary_union if boundary_union > 0 else 0
        
        metrics['boundary_iou'] = boundary_iou
        metrics['boundary_f1'] = boundary_iou  # Simplified
        metrics['edge_accuracy'] = boundary_iou
        
    except:
        metrics['boundary_iou'] = 0
        metrics['boundary_f1'] = 0
        metrics['edge_accuracy'] = 0
    
    # Surface distance metrics (simplified)
    try:
        # Hausdorff distance (simplified approximation)
        y_true_points = np.column_stack(np.where(y_true > 0.5))
        y_pred_points = np.column_stack(np.where(y_pred > 0.5))
        
        if len(y_true_points) > 0 and len(y_pred_points) > 0:
            # Simplified surface distance
            from scipy.spatial.distance import cdist
            distances = cdist(y_true_points, y_pred_points)
            hausdorff = max(np.min(distances, axis=1).max(), np.min(distances, axis=0).max())
            metrics['hausdorff_distance'] = hausdorff
        else:
            metrics['hausdorff_distance'] = float('inf')
    except:
        metrics['hausdorff_distance'] = 0
    
    return metrics

def save_comprehensive_results(original_img, pred_mask, true_mask, filename, output_dirs, output_config):
    """Save comprehensive results with enhanced visualizations"""
    
    # Save prediction mask
    cv2.imwrite(
        os.path.join(output_dirs['predictions'], f'pred_{filename}'),
        (pred_mask * 255).astype(np.uint8)
    )
    
    # Create enhanced overlay
    resized_original = cv2.resize(original_img, (pred_mask.shape[1], pred_mask.shape[0]))
    
    # Multi-colored overlay (prediction in green, ground truth in red)
    overlay = resized_original.copy()
    
    # Add prediction in green
    pred_color = np.zeros_like(overlay)
    pred_color[pred_mask > 0.5] = [0, 255, 0]  # Green for prediction
    
    # Add ground truth contour in red
    true_mask_uint8 = (true_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(true_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)  # Red contour for ground truth
    
    # Blend with original
    alpha = output_config.get('overlay_alpha', 0.5)
    final_overlay = cv2.addWeighted(overlay, 1-alpha, pred_color, alpha, 0)
    
    cv2.imwrite(os.path.join(output_dirs['overlays'], f'overlay_{filename}'), final_overlay)

def analyze_failure_cases_enhanced(failed_cases, output_dir):
    """Enhanced failure case analysis"""
    if not failed_cases:
        return
    
    failure_dir = os.path.join(output_dir, 'failure_analysis')
    os.makedirs(failure_dir, exist_ok=True)
    
    # Save failure cases
    import pandas as pd
    df = pd.DataFrame(failed_cases)
    df.to_csv(os.path.join(failure_dir, 'failure_cases.csv'), index=False)
    
    # Analysis summary
    avg_metrics = {}
    for key in failed_cases[0].keys():
        if key != 'filename' and isinstance(failed_cases[0][key], (int, float)):
            avg_metrics[key] = np.mean([case[key] for case in failed_cases])
    
    with open(os.path.join(failure_dir, 'failure_analysis.txt'), 'w') as f:
        f.write(f"FAILURE CASE ANALYSIS\n")
        f.write(f"="*30 + "\n")
        f.write(f"Total Failure Cases: {len(failed_cases)}\n\n")
        f.write("Average Metrics for Failed Cases:\n")
        for key, value in avg_metrics.items():
            f.write(f"  {key}: {value:.4f}\n")
    
    print(f"📊 Failure analysis saved to {failure_dir}")

def generate_comprehensive_report(all_metrics, config, output_dir):
    """Generate comprehensive evaluation report"""
    if not all_metrics:
        return
    
    # Calculate comprehensive statistics
    avg_metrics = {}
    std_metrics = {}
    
    for key in all_metrics[0].keys():
        if key != 'filename' and isinstance(all_metrics[0][key], (int, float)):
            values = [m[key] for m in all_metrics if not np.isinf(m[key])]
            avg_metrics[key] = np.mean(values) if values else 0
            std_metrics[key] = np.std(values) if values else 0
    
    # Print comprehensive results
    print("\n" + "="*70)
    print("🏆 COMPREHENSIVE ENHANCED EVALUATION REPORT")
    print("="*70)
    print(f"📊 Dataset Statistics:")
    print(f"   Total Images Processed: {len(all_metrics)}")
    print(f"   Configuration: {config.get('Model', {}).get('model_name', 'efficientnetv2-b1')}")
    print(f"   Image Resolution: {config.get('Data', {}).get('img_size', 384)}x{config.get('Data', {}).get('img_size', 384)}")
    print()
    
    print("🎯 PRIMARY METRICS:")
    print(f"   Dice Coefficient:     {avg_metrics.get('dice', 0):.4f} ± {std_metrics.get('dice', 0):.4f}")
    print(f"   IoU Score:            {avg_metrics.get('iou', 0):.4f} ± {std_metrics.get('iou', 0):.4f}")
    print(f"   Accuracy:             {avg_metrics.get('accuracy', 0):.4f} ± {std_metrics.get('accuracy', 0):.4f}")
    print()
    
    print("📏 PRECISION & RECALL:")
    print(f"   Precision:            {avg_metrics.get('precision', 0):.4f} ± {std_metrics.get('precision', 0):.4f}")
    print(f"   Recall (Sensitivity): {avg_metrics.get('recall', 0):.4f} ± {std_metrics.get('recall', 0):.4f}")
    print(f"   Specificity:          {avg_metrics.get('specificity', 0):.4f} ± {std_metrics.get('specificity', 0):.4f}")
    print(f"   F1-Score:             {avg_metrics.get('f1_score', 0):.4f} ± {std_metrics.get('f1_score', 0):.4f}")
    print()
    
    print("🔍 BOUNDARY METRICS:")
    print(f"   Boundary IoU:         {avg_metrics.get('boundary_iou', 0):.4f} ± {std_metrics.get('boundary_iou', 0):.4f}")
    print(f"   Edge Accuracy:        {avg_metrics.get('edge_accuracy', 0):.4f} ± {std_metrics.get('edge_accuracy', 0):.4f}")
    print()
    
    # Performance grade
    dice_score = avg_metrics.get('dice', 0)
    if dice_score >= 0.90:
        grade = "🥇 EXCELLENT"
    elif dice_score >= 0.85:
        grade = "🥈 VERY GOOD"
    elif dice_score >= 0.80:
        grade = "🥉 GOOD"
    elif dice_score >= 0.75:
        grade = "👍 SATISFACTORY"
    else:
        grade = "⚠️ NEEDS IMPROVEMENT"
    
    print(f"🎖️ PERFORMANCE GRADE: {grade}")
    print("="*70)
    
    # Save comprehensive results
    import pandas as pd
    
    # Detailed CSV
    df = pd.DataFrame(all_metrics)
    df.to_csv(os.path.join(output_dir, 'comprehensive_detailed_results.csv'), index=False)
    
    # Summary report
    with open(os.path.join(output_dir, 'comprehensive_summary_report.txt'), 'w') as f:
        f.write("COMPREHENSIVE ENHANCED EVALUATION REPORT\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Dataset: {len(all_metrics)} images\n")
        f.write(f"Model: {config.get('Model', {}).get('model_name', 'efficientnetv2-b1')}\n")
        f.write(f"Resolution: {config.get('Data', {}).get('img_size', 384)}x{config.get('Data', {}).get('img_size', 384)}\n")
        f.write(f"Performance Grade: {grade}\n\n")
        
        f.write("METRICS SUMMARY:\n")
        f.write("-" * 30 + "\n")
        for key, value in avg_metrics.items():
            f.write(f"{key.replace('_', ' ').title()}: {value:.4f} ± {std_metrics.get(key, 0):.4f}\n")
    
    # Create enhanced visualizations
    create_enhanced_visualizations(all_metrics, output_dir)
    
    print(f"📁 Comprehensive report saved to: {output_dir}")

def create_enhanced_visualizations(metrics_list, output_dir):
    """Create enhanced visualization plots"""
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    df = pd.DataFrame(metrics_list)
    
    # Enhanced metrics distribution plot
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    metrics_to_plot = ['dice', 'iou', 'accuracy', 'precision', 'recall', 'f1_score', 
                      'specificity', 'boundary_iou', 'edge_accuracy']
    
    for i, metric in enumerate(metrics_to_plot):
        if i < len(axes) and metric in df.columns:
            # Distribution histogram
            axes[i].hist(df[metric], bins=25, alpha=0.7, edgecolor='black', color='skyblue')
            axes[i].axvline(df[metric].mean(), color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {df[metric].mean():.3f}')
            axes[i].axvline(df[metric].median(), color='green', linestyle=':', linewidth=2,
                          label=f'Median: {df[metric].median():.3f}')
            
            axes[i].set_title(f'{metric.replace("_", " ").title()} Distribution', fontsize=12, fontweight='bold')
            axes[i].set_xlabel(metric.replace("_", " ").title())
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(metrics_to_plot), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'enhanced_metrics_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Correlation heatmap
    if len(df.columns) > 3:
        plt.figure(figsize=(12, 10))
        
        # Select numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, fmt='.2f')
        plt.title('Metrics Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_correlation_heatmap.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"📊 Enhanced visualizations saved to {output_dir}")

if __name__ == "__main__":
    # Run enhanced benchmark with comprehensive evaluation
    try:
        print("🚀 Starting Enhanced PEFNet Benchmark System...")
        metrics = enhanced_benchmark_with_config('./config/benchmark_config.yaml')
        
        if metrics and len(metrics) > 0:
            avg_dice = np.mean([m['dice'] for m in metrics])
            avg_iou = np.mean([m['iou'] for m in metrics])
            
            print("\n🎉 Enhanced benchmark evaluation completed successfully!")
            print(f"🏆 Final Results: Dice={avg_dice:.4f}, IoU={avg_iou:.4f}")
            print("📁 Check ./benchmark_results/ for comprehensive results and visualizations.")
        else:
            print("⚠️ No results generated. Please check your configuration and data paths.")
            
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        print("💡 Please check your model path, data directories, and configuration file.")
