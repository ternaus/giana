import albumentations as albu


def get_train_transform():
    crop_height = 256
    crop_width = 256

    return albu.Compose([
        albu.PadIfNeeded(min_height=crop_height, min_width=crop_width, p=1),
        albu.RandomSizedCrop((int(0.3 * crop_height), 288), crop_height, crop_width, p=1),
        albu.ShiftScaleRotate(rotate_limit=45, p=1),
        albu.HueSaturationValue(p=0.5),
        albu.RandomGamma(p=1),
        albu.RandomBrightnessContrast(p=1, brightness_limit=0.5, contrast_limit=0.5),
        albu.HorizontalFlip(p=0.5),
        albu.Normalize(p=1)
    ], p=1)


def get_val_transform():
    crop_height = 288
    crop_width = 320

    return albu.Compose([
        albu.PadIfNeeded(min_height=crop_height, min_width=crop_width, p=1),
        albu.Normalize(p=1)
    ], p=1)
