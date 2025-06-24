
import tensorflow as tf
from model import build_model
from metrics.segmentation_metrics import dice_coeff, bce_dice_loss, IoU, zero_IoU, dice_loss
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import get_custom_objects
from sklearn.model_selection import train_test_split
from callbacks.callbacks import get_callbacks
from dataloader.dataloader import build_augmenter, build_dataset, build_decoder
import yaml

def main(config_file):
    try:
        # Đọc file cấu hình
        print("Reading config file:", config_file)
        with open(config_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        print("Parameter for Training:", data)

        # Lấy tham số
        img_size = data['Data']['img_size']
        route = os.path.normpath(os.path.abspath(data['Data']['route']))  # Xử lý ký tự tiếng Việt
        X_path = data['Data']['X_path']
        Y_path = data['Data']['Y_path']
        images_dir = os.path.join(route, X_path)
        masks_dir = os.path.join(route, Y_path)

        valid_size = data['Hyperparameter']['valid_size']
        test_size = data['Hyperparameter']['test_size']
        SEED = data['Hyperparameter']['SEED']
        BATCH_SIZE = data['Hyperparameter']['BATCH_SIZE']
        epochs = data['Hyperparameter']['epochs']
        max_lr = data['Hyperparameter']['max_lr']
        min_lr = data['Hyperparameter']['min_lr']
        save_weights_only = data['Hyperparameter']['save_weights_only']
        save_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(config_file)), data['Model']['save_path']))

        # Kiểm tra và tạo thư mục checkpoints nếu cần
        checkpoint_dir = os.path.dirname(save_path)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            print(f"Created checkpoint directory: {checkpoint_dir}")

        # Kiểm tra thư mục dữ liệu
        print("Checking directories...")
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not os.path.exists(masks_dir):
            raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

        print("LOAD DATA:")
        # Liệt kê file
        X_full = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
        Y_full = sorted([f for f in os.listdir(masks_dir) if f.endswith('.jpg')])

        # Kiểm tra đồng bộ
        if len(X_full) == 0 or len(Y_full) == 0:
            raise ValueError("No .jpg files found in images/ or masks/")
        if len(X_full) != len(Y_full):
            raise ValueError(f"Mismatch: {len(X_full)} images, {len(Y_full)} masks")
        if set(X_full) != set(Y_full):
            raise ValueError("Image and mask filenames do not match")

        # Tạo đường dẫn đầy đủ
        X_full_paths = [os.path.join(images_dir, f) for f in X_full]
        Y_full_paths = [os.path.join(masks_dir, f) for f in Y_full]

        # Kiểm tra file tồn tại
        for img_path, mask_path in zip(X_full_paths, Y_full_paths):
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask file not found: {mask_path}")

        print(f"Loaded {len(X_full)} images and {len(Y_full)} masks")

        # Chia tập dữ liệu
        X_train, X_valid = train_test_split(X_full_paths, test_size=valid_size, random_state=SEED)
        Y_train, Y_valid = train_test_split(Y_full_paths, test_size=valid_size, random_state=SEED)
        X_train, X_test = train_test_split(X_train, test_size=test_size, random_state=SEED)
        Y_train, Y_test = train_test_split(Y_train, test_size=test_size, random_state=SEED)

        print("N Train:", len(X_train))
        print("N Valid:", len(X_valid))
        print("N Test:", len(X_test))

        print("BUILD DATASETS:")
        train_decoder = build_decoder(with_labels=True, target_size=(img_size, img_size), ext='jpg', segment=True, ext2='jpg')
        train_dataset = build_dataset(X_train, Y_train, bsize=BATCH_SIZE, decode_fn=train_decoder,
                                     augmentAdv=False, augment=False, augmentAdvSeg=True)

        valid_decoder = build_decoder(with_labels=True, target_size=(img_size, img_size), ext='jpg', segment=True, ext2='jpg')
        valid_dataset = build_dataset(X_valid, Y_valid, bsize=BATCH_SIZE, decode_fn=valid_decoder,
                                     augmentAdv=False, augment=False, repeat=False, shuffle=False,
                                     augmentAdvSeg=False)

        test_decoder = build_decoder(with_labels=True, target_size=(img_size, img_size), ext='jpg', segment=True, ext2='jpg')
        test_dataset = build_dataset(X_test, Y_test, bsize=BATCH_SIZE, decode_fn=test_decoder,
                                    augmentAdv=False, augment=False, repeat=False, shuffle=False,
                                    augmentAdvSeg=False)

        print("BUILD MODEL:")
        model = build_model(img_size)
        model.summary()

        get_custom_objects().update({"dice": dice_loss})
        model.compile(optimizer=Adam(learning_rate=1e-3),
                      loss='dice',
                      metrics=[dice_coeff, bce_dice_loss, IoU, zero_IoU])

        callbacks = get_callbacks(monitor='val_loss', mode='min', save_path=save_path,
                                 max_lr=max_lr, min_lr=min_lr, cycle_epoch=1000,
                                 save_weights_only=save_weights_only)

        steps_per_epoch = len(X_train) // BATCH_SIZE

        print("START TRAINING:")
        his = model.fit(train_dataset,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=valid_dataset)

        print("Training completed!")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise
    except KeyError as e:
        print(f"Error: Missing key {e} in train_config.yaml")
        raise
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML format in {config_file}: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

if __name__ == "__main__":
    config_file = "./config/train_config.yaml"
    main(config_file)