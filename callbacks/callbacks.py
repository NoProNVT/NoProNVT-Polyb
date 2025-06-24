import math
import tensorflow as tf
import matplotlib.pyplot as plt

# Hàm cosine annealing có warm-up (dùng cho LearningRateScheduler)
def cosine_annealing_with_warmup(epochIdx):
    aMax, aMin = max_lr, min_lr
    warmupEpochs, stagnateEpochs, cosAnnealingEpochs = 0, 0, cycle_epoch
    epochIdx = epochIdx % (warmupEpochs + stagnateEpochs + cosAnnealingEpochs)
    
    if epochIdx < warmupEpochs:
        return aMin + (aMax - aMin) / (warmupEpochs - 1) * epochIdx
    else:
        epochIdx -= warmupEpochs

    if epochIdx < stagnateEpochs:
        return aMax
    else:
        epochIdx -= stagnateEpochs

    return aMin + 0.5 * (aMax - aMin) * (1 + math.cos((epochIdx + 1) / (cosAnnealingEpochs + 1) * math.pi))

# Hàm vẽ biểu đồ learning rate theo epoch
def plt_lr(step, schedulers):
    x = range(step)
    y = [schedulers(_) for _ in x]

    plt.plot(x, y)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Cosine Annealing Schedule')
    plt.legend()
    plt.show()

# Hàm trả về list các callback dùng khi train
def get_callbacks(monitor, mode, save_path, max_lr, min_lr, cycle_epoch, save_weights_only):
    # Lưu các giá trị này thành biến toàn cục để hàm cosine_annealing_with_warmup có thể dùng
    globals()['max_lr'] = max_lr
    globals()['min_lr'] = min_lr
    globals()['cycle_epoch'] = cycle_epoch

    #Callback: Dừng sớm khi val_loss không cải thiện
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
       patience=10,
       mode=mode
    )

    # Callback: Giảm LR khi val_loss không cải thiện
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=0.2,
        patience=5,
        verbose=1,
        mode=mode,
        min_lr=1e-5,
    )

    # Callback: Lưu model tốt nhất
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_path,
        monitor=monitor,
        verbose=1,
        save_best_only=True,
        save_weights_only=save_weights_only,
        mode=mode,
        save_freq="epoch",
    )

    # Callback: Điều chỉnh learning rate theo lịch cosine
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(cosine_annealing_with_warmup, verbose=0)

    # Callback: Ghi log quá trình huấn luyện vào file CSV
    csv_logger = tf.keras.callbacks.CSVLogger('training.csv')

    # Trả về list callback
    callbacks = [checkpoint, lr_schedule, csv_logger]
    return callbacks