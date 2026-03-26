import segmentation_models_pytorch as smp

def build_deeplabv3plus(
    num_classes: int,
    in_channels: int = 5,
    encoder_weights: str = "imagenet",
    encoder_name: str = "resnet50",
    ):

    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes
    )
    
    return model
