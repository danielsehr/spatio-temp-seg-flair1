from itertools import islice
import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from scipy.ndimage import binary_erosion

from utils.albumentation import transform


#--- Color mapping ---#
num_classes = 19

class_cmap = {
1   : ['building','#db0e9a'] ,
2   : ['pervious surface','#938e7b'],
3   : ['impervious surface','#f80c00'],
4   : ['bare soil','#a97101'],
5   : ['water','#1553ae'],
6   : ['coniferous','#194a26'],
7   : ['deciduous','#46e483'],
8   : ['brushwood','#f3a60d'],
9   : ['vineyard','#660082'],
10  : ['herbaceous vegetation','#55ff00'],
11  : ['agricultural land','#fff30d'],
12  : ['plowed land','#e4df7c'],
13  : ['swimming_pool','#3de6eb'],
14  : ['snow','#ffffff'],
15  : ['clear cut','#8ab3a0'],
16  : ['mixed','#6b714f'],
17  : ['ligneous','#c5dc42'],
18  : ['greenhouse','#9999ff'],
19  : ['other','#000000'],
}

colors = [class_cmap[i][1] for i in class_cmap.keys()]

color_map_flair = ListedColormap(np.array(colors))

bounds = np.arange(0.5, num_classes + 1.5, 1.0)
seg_norm = BoundaryNorm(bounds, color_map_flair.N)


# --- Plotting ---#
def denormalize(
    image: torch.tensor,
    mean: list = [105.08, 110.87, 101.82, 106.38, 53.26],
    std: list = [52.17, 45.38, 44.00, 39.69, 79.30],
    max_pixel_value: float = 1.0    # 
    ) -> torch.tensor:
    
    assert len(mean) == len(std)
    assert image.shape[0] == 3

    # Denormalize image
    mean = torch.tensor(mean[:3]).view(3,1,1)
    std = torch.tensor(std[:3]).view(3,1,1)
    
    image = (image * std * max_pixel_value) + mean * max_pixel_value
    image = image / 255.0

    return image


def plot_image_mask(
    data_loader: DataLoader,
    take: int,
    num_col: int = 4,
    normalized: bool = True,
    verbose: bool = False
    ) -> None:
    
    image_list = []
    mask_list = []

    # Set batchsize depending on number of desired images
    batchsize: int = data_loader.batch_size
    
    num_batches = take // batchsize
    if num_batches == 0:
        num_batches = 1
    assert num_batches != 0, "Number of batches = 0"


    # Load data into lists
    
    for dic in islice(data_loader, num_batches):
        for image, mask in zip(dic["image"], dic["mask"]):

            if image.shape[0] > 3:
                image = image[:3, ...]
            
            # Denormalize image from imagenet weights
            if normalized:
                image = denormalize(image)

            image = image.permute(1, 2, 0)

            image_list.append(image)
            mask_list.append(mask)

            if verbose==True:
                print(f"Image shape: {image.shape}")
                print(f"Mask shape: {mask.shape}\n")

    # Define numbers of rows and cols to plot
    if take == 1:
        cols = 2
        rows = 1
    else:
        cols = num_col
        total = take*2

        if total % cols != 0:
            rows = (total // cols) + 1
        else:
            rows = (total // cols)


    # Plot
    fig, axs = plt.subplots(figsize=(10,10), ncols = cols, nrows = rows)
    axs = axs.flatten()

    for i, n in zip(range(0, cols*rows, 2), range(0, take, 1)):
        axs[i].imshow(image_list[n])
        axs[i].axis("off")
        axs[i].set_title("rgb")

    for j, n in zip(range(1, cols*rows, 2), range(0, take, 1)):
        axs[j].imshow(mask_list[n], cmap="tab20")
        axs[j].axis("off")
        axs[j].set_title("mask")

    plt.tight_layout()
    plt.show()
    

    
def plot_triplet_figure(
    image: np.ndarray,
    mask: np.ndarray,
    pred: np.ndarray,
    iou: np.ndarray,
    num_classes: int,
    class_cmap: ListedColormap,
    labels: list
    ):

    image = image.permute(1, 2, 0).cpu().numpy()
    mask = mask.cpu().numpy()
    pred = pred.cpu().numpy()
    iou = iou.cpu().numpy()

    # Ground Truth classes
    classes_gt = np.unique(mask)
    classes_gt = classes_gt[(classes_gt > 0) & (classes_gt <= num_classes)].astype(int)
    
    # Prediction classes
    classes_pred = np.unique(pred)
    classes_pred = classes_pred[(classes_pred >= 0) & (classes_pred <= num_classes)].astype(int)

    # Contour line functions
    def draw_contour(
        mask: np.ndarray, 
        class_id: int,
        class_cmap
        ) -> np.ndarray:

        color = class_cmap.colors[class_id]

        class_mask = (mask == class_id).astype(float)
        if not np.any(class_mask):
            pass

        inner_mask = binary_erosion(class_mask, iterations=2)

        return inner_mask, color



    # --- PLOT
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11, 9))

    ax_img  = axes[0, 0]
    ax_gt   = axes[0, 1]
    ax_ovl  = axes[1, 0]
    ax_pred = axes[1, 1]

    # Contour settings
    contour_kwargs = dict(
        levels=[0.5],
        linewidths=1.2,
    )

    legend_kwargs = dict(
        title="Predicted classes",
        loc="center right",
        bbox_to_anchor=(1.5, 0.5),
    )


    # -- IMAGE AXES
    ax_img.imshow(image)
    ax_img.set_title("Image + Mask Contours")
    ax_img.axis("off")
    
    # draw contour & labels
    legend_patches_img = []

    for class_id in classes_gt:
        class_mask, color = draw_contour(mask=mask, class_id=class_id, class_cmap=class_cmap)
        ax_img.contour(
            class_mask,
            colors=[color],
            **contour_kwargs
        )

        legend_patches_img.append(
            Patch(facecolor=color, edgecolor="black", label=labels[class_id])
        )

        if legend_patches_img:
            ax_img.legend(
                handles=legend_patches_img,
                **legend_kwargs
            )


    # -- GT AXES
    ax_gt.imshow(mask, cmap=class_cmap, norm=class_norm)
    ax_gt.set_title("Ground Truth")
    ax_gt.axis("off")    



    # -- OVERLAY AXES
    ax_ovl.imshow(image)
    ax_ovl.set_title("Image + Prediction Contours")
    ax_ovl.axis("off")

    # draw contour and legend
    legend_patches_ovl = []

    for class_id in classes_pred:
        class_mask, color = draw_contour(mask=pred, class_id=class_id, class_cmap=class_cmap)
        ax_ovl.contour(
            class_mask,
            colors=[color],
            **contour_kwargs
        )

        legend_patches_ovl.append(
            Patch(facecolor=color, edgecolor="black", label=labels[class_id])
        )

        if legend_patches_ovl:
            ax_ovl.legend(
                handles=legend_patches_ovl,
                **legend_kwargs
            )


    # -- PREDICTION AXES
    ax_pred.imshow(pred, cmap=class_cmap, norm=class_norm)
    ax_pred.set_title(f"Prediction | IoU: {iou:.3f}")
    ax_pred.axis("off")

    fig.set_constrained_layout(True)