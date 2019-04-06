import albumentations as albu


def get_train_transform():
    crop_height = 256
    crop_width = 256

    return albu.Compose([
        albu.PadIfNeeded(min_height=crop_height, min_width=crop_width, p=1),
        albu.RandomSizedCrop((int(0.5 * crop_height), 288), crop_height, crop_width, p=1),
        albu.RandomCrop(height=crop_height, width=crop_width, p=1),
        albu.VerticalFlip(p=0.5),
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
