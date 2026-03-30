import albumentations as A
import numpy as np
import cv2


transform = A.Compose([
    A.Normalize(
        mean = (105.08, 110.87, 101.82, 106.38, 53.26),
        std  = (52.17, 45.38, 44.00, 39.69, 79.30),
        max_pixel_value=1.0,
        p=1.0
    )
])



# Augment parameters
crop_scales = [0.9, 0.8, 0.7]
crop_sizes = [int(512 * s) for s in crop_scales]
final_size = int(512)

# # Helper to apply gamma only on rgb channels
# def gamma_rgb_only(image, **kwargs):
#     rgb = image[..., :3]
#     extra = image[..., 3:]

#     rgb_aug = A.RandomGamma(gamma_limit=(60, 150), p=1.0)(image=rgb)["image"] #40 
    
#     return np.concatenate([rgb_aug, extra], axis=-1)


augment = A.Compose(
    [
        # #--- Photometric ---
        # A.RandomBrightnessContrast(
        #     brightness_limit=(-0.2, 0.2),
        #     contrast_limit=(-0.2, 0.2),
        #     p=0.5,
        # ),

        # A.Lambda(
        #     name="select_rgb_gamma", 
        #     image=gamma_rgb_only, p=0.5
        #     ),

        # A.GaussNoise(
        #     std_range=(0.02, 0.06),
        #     p=0.5,
        # ),

        # A.GaussianBlur(
        #     blur_limit=(1, 3),
        #     p=0.5,
        # ),


        #--- Geometric ---
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(
            limit=45,
            p=0.5,
            border_mode=cv2.BORDER_REFLECT,
            interpolation=cv2.INTER_CUBIC,
            mask_interpolation=cv2.INTER_NEAREST
            ),
        A.OneOf(
            [A.RandomCrop(size, size) for size in crop_sizes],
            p=0.6
        ),
        A.Resize(
            final_size, final_size, 
            interpolation=cv2.INTER_CUBIC,
            mask_interpolation=cv2.INTER_NEAREST
            ),
    ],
)