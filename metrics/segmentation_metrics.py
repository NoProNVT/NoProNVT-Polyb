import tensorflow as tf 
import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy

def dice_coeff(y_true, y_pred, smooth=1e-7):
    """Improved dice coefficient with numerical stability"""
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice_score = (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    
    return dice_score

def dice_loss(y_true, y_pred):
    """Dice loss function"""
    return 1.0 - dice_coeff(y_true, y_pred)

def IoU(y_true, y_pred, smooth=1e-7):
    """Intersection over Union with improved numerical stability"""
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    
    iou_score = (intersection + smooth) / (union + smooth)
    return iou_score

def zero_IoU(y_true, y_pred):
    """IoU for background (negative class)"""
    return IoU(1-y_true, 1-y_pred)

def bce_dice_loss(y_true, y_pred, bce_weight=0.5, dice_weight=0.5):
    """Combined Binary Cross Entropy and Dice Loss"""
    bce = binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce_weight * bce + dice_weight * dice

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """Focal loss for handling class imbalance"""
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    
    # Calculate focal loss
    cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    
    focal_loss_value = alpha_t * tf.pow((1 - p_t), gamma) * cross_entropy
    
    return tf.reduce_mean(focal_loss_value)

def focal_dice_loss(y_true, y_pred, focal_weight=0.7, dice_weight=0.3):
    """Combined Focal and Dice Loss"""
    focal = focal_loss(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return focal_weight * focal + dice_weight * dice

def tversky(y_true, y_pred, alpha=0.7, smooth=1e-7):
    """Tversky index - generalization of dice coefficient"""
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    
    true_pos = tf.reduce_sum(y_true_f * y_pred_f)
    false_neg = tf.reduce_sum(y_true_f * (1 - y_pred_f))
    false_pos = tf.reduce_sum((1 - y_true_f) * y_pred_f)
    
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

def tversky_loss(y_true, y_pred, alpha=0.7):
    """Tversky loss function"""
    return 1.0 - tversky(y_true, y_pred, alpha)

def focal_tversky_loss(y_true, y_pred, alpha=0.7, gamma=0.75):
    """Focal Tversky Loss for better handling of hard examples"""
    tv = tversky(y_true, y_pred, alpha)
    return tf.pow((1 - tv), gamma)

def combo_loss(y_true, y_pred, alpha=0.5, beta=0.5, gamma=2.0):
    """Combination of Focal and Dice losses with adjustable weights"""
    focal = focal_loss(y_true, y_pred, gamma=gamma)
    dice = dice_loss(y_true, y_pred)
    return alpha * focal + beta * dice

def boundary_loss(y_true, y_pred):
    """Boundary-aware loss for better edge detection"""
    # Sobel filters for edge detection
    sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
    sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tf.float32)
    
    sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
    sobel_y = tf.reshape(sobel_y, [3, 3, 1, 1])
    
    # Calculate gradients
    grad_x_true = tf.nn.conv2d(y_true, sobel_x, strides=[1, 1, 1, 1], padding='SAME')
    grad_y_true = tf.nn.conv2d(y_true, sobel_y, strides=[1, 1, 1, 1], padding='SAME')
    grad_true = tf.sqrt(tf.square(grad_x_true) + tf.square(grad_y_true))
    
    grad_x_pred = tf.nn.conv2d(y_pred, sobel_x, strides=[1, 1, 1, 1], padding='SAME')
    grad_y_pred = tf.nn.conv2d(y_pred, sobel_y, strides=[1, 1, 1, 1], padding='SAME')
    grad_pred = tf.sqrt(tf.square(grad_x_pred) + tf.square(grad_y_pred))
    
    # Calculate boundary loss
    boundary_diff = tf.square(grad_true - grad_pred)
    return tf.reduce_mean(boundary_diff)

def dice_boundary_loss(y_true, y_pred, dice_weight=0.8, boundary_weight=0.2):
    """Combined Dice and Boundary loss"""
    dice = dice_loss(y_true, y_pred)
    boundary = boundary_loss(y_true, y_pred)
    return dice_weight * dice + boundary_weight * boundary

# Additional metrics for comprehensive evaluation
def precision(y_true, y_pred, threshold=0.5):
    """Precision metric"""
    y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
    y_true_binary = tf.cast(y_true > threshold, tf.float32)
    
    true_pos = tf.reduce_sum(y_true_binary * y_pred_binary)
    pred_pos = tf.reduce_sum(y_pred_binary)
    
    return tf.where(pred_pos == 0, 0.0, true_pos / pred_pos)

def recall(y_true, y_pred, threshold=0.5):
    """Recall metric"""
    y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
    y_true_binary = tf.cast(y_true > threshold, tf.float32)
    
    true_pos = tf.reduce_sum(y_true_binary * y_pred_binary)
    actual_pos = tf.reduce_sum(y_true_binary)
    
    return tf.where(actual_pos == 0, 0.0, true_pos / actual_pos)

def f1_score(y_true, y_pred, threshold=0.5):
    """F1-score metric"""
    prec = precision(y_true, y_pred, threshold)
    rec = recall(y_true, y_pred, threshold)
    
    return tf.where((prec + rec) == 0, 0.0, 2 * (prec * rec) / (prec + rec))

def sensitivity(y_true, y_pred, threshold=0.5):
    """Sensitivity (same as recall)"""
    return recall(y_true, y_pred, threshold)

def specificity(y_true, y_pred, threshold=0.5):
    """Specificity metric"""
    y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
    y_true_binary = tf.cast(y_true > threshold, tf.float32)
    
    true_neg = tf.reduce_sum((1 - y_true_binary) * (1 - y_pred_binary))
    actual_neg = tf.reduce_sum(1 - y_true_binary)
    
    return tf.where(actual_neg == 0, 0.0, true_neg / actual_neg)

# Custom metrics for model compilation
def mean_iou(y_true, y_pred):
    """Mean IoU across batch"""
    return IoU(y_true, y_pred)

def mean_dice(y_true, y_pred):
    """Mean Dice coefficient across batch"""
    return dice_coeff(y_true, y_pred)

# Loss function recommendations for different scenarios
def get_loss_function(loss_type='combo'):
    """Get recommended loss function based on type"""
    loss_functions = {
        'dice': dice_loss,
        'focal': focal_loss,
        'tversky': tversky_loss,
        'focal_tversky': focal_tversky_loss,
        'bce_dice': bce_dice_loss,
        'focal_dice': focal_dice_loss,
        'combo': combo_loss,
        'dice_boundary': dice_boundary_loss
    }
    
    return loss_functions.get(loss_type, combo_loss)

def get_metrics():
    """Get comprehensive list of metrics for training"""
    return [
        mean_dice,
        mean_iou,
        precision,
        recall,
        f1_score,
        sensitivity,
        specificity
    ]
