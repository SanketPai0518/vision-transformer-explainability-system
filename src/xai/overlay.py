import numpy as np
import cv2

def overlay_heatmap(image, heatmap, alpha=0.5):
    """
    image: PIL image resized to 224x224
    heatmap: (224,224) normalized
    """
    img = np.array(image.resize((224, 224)))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

    return overlay
