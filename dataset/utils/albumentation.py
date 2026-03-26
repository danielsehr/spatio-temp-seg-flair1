import albumentations as A
import cv2

crop_size = int(256*0.8)
final_size = int(256)

augment = A.Compose(
    [
        # Geometric
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(
            limit=45,
            p=0.5,
            border_mode=cv2.BORDER_REFLECT,
            interpolation=cv2.INTER_CUBIC,
            mask_interpolation=cv2.INTER_NEAREST
            ),
        A.RandomCrop(crop_size, crop_size, p=0.5, border_mode=cv2.BORDER_REFLECT),
        A.Resize(
            final_size, final_size, 
            interpolation=cv2.INTER_CUBIC,
            mask_interpolation=cv2.INTER_NEAREST
            ),
    ],
    seed=42
)