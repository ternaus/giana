import segmentation_models_pytorch as smp


def get_model(model_name):
    return smp.Unet(model_name, encoder_weights='imagenet', activation=None)
