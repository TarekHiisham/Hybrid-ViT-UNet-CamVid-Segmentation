"""
CamVid dataset class definitions and colormap.
32 semantic classes used in training.
"""

import numpy as np

# CamVid 32-class definitions: (name, R, G, B)
CAMVID_CLASSES = [
("Animal", 64,128,64),            
("Archway",192,0,128),
("Bicyclist",0,128,192),
("Bridge",0,128,64),
("Building",128,0,0),
("Car",64,0,128),
("CartLuggagePram",64,0,192),
("Child",192,128,64),
("Column_Pole",192,192,128),
("Fence",64,64,128),
("LaneMkgsDriv",128,0,192),
("LaneMkgsNonDriv",192,0,64),
("Misc_Text",128,128,64),
("MotorcycleScooter",192,0,192),
("OtherMoving",128,64,64),
("ParkingBlock",64,192,128),
("Pedestrian",64,64,0),
("Road",128,64,128),
("RoadShoulder",128,128,192),
("Sidewalk",0,0,192),
("SignSymbol",192,128,128),
("Sky",128,128,128),
("SUVPickupTruck",64,128,192),
("TrafficCone",0,0,64),
("TrafficLight",0,64,64),
("Train",192,64,128),
("Tree",128,128,0),
("Truck_Bus",192,128,192),
("Tunnel",64,0,64),
("VegetationMisc",192,192,0),
("Void",0,0,0),
("Wall",64,192,0)
]

# Build a 32-color palette array: shape (32, 3)
PALETTE = np.array([c[1:] for c in CAMVID_CLASSES], dtype=np.uint8)

NUM_CLASSES = 32


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    """
    Convert integer class mask [H, W] to RGB color image [H, W, 3].
    Values outside [0, NUM_CLASSES-1] map to black.
    """
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx in range(NUM_CLASSES):
        color_img[mask == cls_idx] = PALETTE[cls_idx]
    return color_img


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Blend segmentation color mask over the original image.

    Args:
        image : uint8 RGB [H, W, 3]  – original, already at DISPLAY size
        mask  : int   [H, W]         – class indices, already at display size
        alpha : float                 – opacity of the segmentation layer (0=invisible, 1=full)

    Returns:
        blended uint8 RGB [H, W, 3]
    """
    color_mask = mask_to_color(mask)          # [H, W, 3]
    blended = (
        (1 - alpha) * image.astype(np.float32)
        + alpha * color_mask.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)
    return blended
