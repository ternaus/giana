import albumentations as albu


def get_train_transform():
    crop_height = 256
    crop_width = 256

    return albu.Compose([
        albu.PadIfNeeded(min_height=crop_height, min_width=crop_width, p=1),
        albu.RandomSizedCrop((int(0.3 * crop_height), 288), crop_height, crop_width, p=1),
        albu.HorizontalFlip(p=0.5),
        albu.OneOf([
            albu.IAAAdditiveGaussianNoise(p=0.5),
            albu.GaussNoise(p=0.5),
        ], p=0.2),
        albu.OneOf([
            albu.MotionBlur(p=0.2),
            albu.MedianBlur(blur_limit=3, p=0.1),
            albu.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0, rotate_limit=20, p=0.1),
        albu.OneOf([
            albu.OpticalDistortion(p=0.3),
            albu.GridDistortion(p=0.1),
            albu.IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        albu.OneOf([
            albu.CLAHE(clip_limit=2, p=0.5),
            albu.IAASharpen(p=0.5),
            albu.IAAEmboss(p=0.5),
            albu.RandomBrightnessContrast(p=0.5),
        ], p=0.3),
        albu.HueSaturationValue(p=0.3),
        albu.JpegCompression(p=0.2, quality_lower=20, quality_upper=99),
        albu.ElasticTransform(p=0.1),
        albu.Normalize(p=1)
    ], p=1)


def get_val_transform():
    crop_height = 288
    crop_width = 320

    return albu.Compose([
        albu.PadIfNeeded(min_height=crop_height, min_width=crop_width, p=1),
        albu.Normalize(p=1)
    ], p=1)
