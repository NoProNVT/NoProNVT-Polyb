import tensorflow as tf
import os 

def auto_select_accelerator():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except ValueError:
        strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy

def default_augment_seg(input_image, input_mask):
    """Enhanced segmentation augmentation with more variety"""
    # Color augmentations
    input_image = tf.image.random_brightness(input_image, 0.15)
    input_image = tf.image.random_contrast(input_image, 0.85, 1.15)
    input_image = tf.image.random_saturation(input_image, 0.85, 1.15)
    input_image = tf.image.random_hue(input_image, 0.02)
    
    # Geometric augmentations
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_up_down(input_image)
        input_mask = tf.image.flip_up_down(input_mask)
    
    # Random rotation (0, 90, 180, 270 degrees)
    k = tf.random.uniform((), maxval=4, dtype=tf.int32)
    input_image = tf.image.rot90(input_image, k)
    input_mask = tf.image.rot90(input_mask, k)
    
    # Add slight Gaussian noise
    if tf.random.uniform(()) > 0.7:
        noise = tf.random.normal(tf.shape(input_image), stddev=0.01)
        input_image = tf.clip_by_value(input_image + noise, 0.0, 1.0)
    
    return input_image, input_mask

def BatchAdvAugment(images, labels):
    """Advanced batch augmentation for classification"""
    return default_augment_seg(images, labels)

def BatchAdvAugmentSeg(images, masks):
    """Advanced batch augmentation for segmentation"""
    return default_augment_seg(images, masks)

@tf.function
def mixup_augment(images, masks, alpha=0.2):
    """Mixup augmentation for better generalization"""
    batch_size = tf.shape(images)[0]
    
    # Generate lambda from beta distribution
    lam = tf.random.gamma([batch_size], alpha, alpha)
    lam = tf.clip_by_value(lam, 0.0, 1.0)
    lam = tf.reshape(lam, [batch_size, 1, 1, 1])
    
    # Shuffle indices
    indices = tf.random.shuffle(tf.range(batch_size))
    
    # Mix images and masks
    mixed_images = lam * images + (1 - lam) * tf.gather(images, indices)
    mixed_masks = lam * masks + (1 - lam) * tf.gather(masks, indices)
    
    return mixed_images, mixed_masks

def build_decoder(with_labels=True, target_size=(256, 256), ext='png', segment=False, ext2='png'):
    def decode(path):
        file_bytes = tf.io.read_file(path)
        if ext == 'png':
            img = tf.image.decode_png(file_bytes, channels=3)
        elif ext in ['jpg', 'jpeg']:
            img = tf.image.decode_jpeg(file_bytes, channels=3)
        else:
            raise ValueError("Image extension not supported")
        
        img = tf.image.resize(img, target_size, method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(img, tf.float32) / 255.0
        return img
    
    def decode_mask(path, gray=True):
        file_bytes = tf.io.read_file(path)
        if ext2 == 'png':
            img = tf.image.decode_png(file_bytes, channels=3)
        elif ext2 in ['jpg', 'jpeg']:
            img = tf.image.decode_jpeg(file_bytes, channels=3)
        elif ext2 in ['tif', 'tiff']:
            img = tf.io.decode_raw(file_bytes, tf.uint8)
            img = tf.reshape(img, [-1, -1, 3])
        else:
            raise ValueError("Mask extension not supported")
        
        if gray:
            img = tf.image.rgb_to_grayscale(img)
        img = tf.image.resize(img, target_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        img = tf.cast(img, tf.float32) / 255.0
        # Ensure binary mask
        img = tf.cast(img > 0.5, tf.float32)
        return img

    def decode_with_labels(path, label):
        return decode(path), label
    
    def decode_with_segments(path, path2, gray=True):
        return decode(path), decode_mask(path2, gray)
    
    if segment:
        return decode_with_segments
    return decode_with_labels if with_labels else decode

def build_augmenter(with_labels=True, advanced=True):
    def augment(img):
        # Basic augmentations
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.1)
        img = tf.image.random_contrast(img, 0.9, 1.1)
        img = tf.image.random_saturation(img, 0.9, 1.1)
        img = tf.image.random_hue(img, 0.02)
        
        if advanced:
            # Advanced augmentations
            if tf.random.uniform(()) > 0.5:
                img = tf.image.random_jpeg_quality(img, 75, 100)
            
            # Random rotation
            k = tf.random.uniform((), maxval=4, dtype=tf.int32)
            img = tf.image.rot90(img, k)
        
        return img
    
    def augment_with_labels(img, label):
        return augment(img), label
    
    return augment_with_labels if with_labels else augment

def build_dataset(paths, labels=None, bsize=32, cache=True,
                  decode_fn=None, augment_fn=None,
                  augment=True, augmentAdv=False, augmentAdvSeg=False, 
                  repeat=True, shuffle=1024, cache_dir="",
                  use_mixup=False, prefetch_size=None):
    
    if cache_dir != "" and cache is True:
        os.makedirs(cache_dir, exist_ok=True)
    
    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)
    
    if augment_fn is None:
        augment_fn = build_augmenter(labels is not None)
    
    AUTO = tf.data.experimental.AUTOTUNE
    if prefetch_size is None:
        prefetch_size = AUTO
        
    slices = paths if labels is None else (paths, labels)
    
    dset = tf.data.Dataset.from_tensor_slices(slices)
    dset = dset.map(decode_fn, num_parallel_calls=AUTO, deterministic=False)
    dset = dset.cache(cache_dir) if cache else dset
    
    if shuffle:
        dset = dset.shuffle(shuffle, reshuffle_each_iteration=True)
    
    dset = dset.map(augment_fn, num_parallel_calls=AUTO) if augment else dset
    dset = dset.repeat() if repeat else dset
    dset = dset.batch(bsize, drop_remainder=True)
    
    # Advanced augmentations
    if augmentAdv:
        dset = dset.map(BatchAdvAugment, num_parallel_calls=AUTO)
    if augmentAdvSeg:
        dset = dset.map(BatchAdvAugmentSeg, num_parallel_calls=AUTO)
    
    # Mixup augmentation
    if use_mixup:
        dset = dset.map(lambda x, y: mixup_augment(x, y), num_parallel_calls=AUTO)
    
    dset = dset.prefetch(prefetch_size)
    
    return dset
