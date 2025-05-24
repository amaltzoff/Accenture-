import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy import ndimage
from skimage import morphology
from skimage.segmentation import clear_border


# Create a binary mask of pixels close in color to the given center within a radius.
def color_mask(image, center, radius):
  
# Compute squared Euclidean distance from each pixel's RGB value to the target color,
# then return a binary mask where True indicates pixels within the specified color radius.
    return np.sum((image - center) ** 2, axis=2) < radius ** 2


# Clean binary mask by filling holes, removing small objects and holes, and clearing borders.
def clean_mask(mask, min_size=4000, hole_size=2000):

    mask = ndimage.binary_fill_holes(mask)
    mask = morphology.remove_small_objects(mask, min_size) 
    mask = morphology.remove_small_holes(mask, hole_size) 
    mask = clear_border(mask)
    return mask


# Label connected regions in a mask and return their centroid coordinates.
def get_centroids(mask):

    x_coords, y_coords = [], []
    labels = ndimage.label(mask)

    for label_id in range(1, labels[1] + 1):  # Skip background label 0
        area = np.sum(labels[0] == label_id)
        if area > 5:
            y, x = ndimage.center_of_mass(mask, labels[0], label_id)
            x_coords.append(int(x))
            y_coords.append(int(y))
    return np.column_stack((x_coords, y_coords))


# Process the image to detect green and brown skittles, count them, and extract centroid coordinates.
def process(image, green_ref, brown_ref):

    # Create and clean masks
    gmask = clean_mask(color_mask(image, green_ref["center"], green_ref["radius"]))
    bmask = clean_mask(color_mask(image, brown_ref["center"], brown_ref["radius"]))

    # Extract centroids
    green_coords = get_centroids(gmask)
    brown_coords = get_centroids(bmask)

    # Output results
    print(f"Detected {len(green_coords)} green skittles at:")
    for x, y in green_coords:
        print(f"  - ({x}, {y})")

    print(f"Detected {len(brown_coords)} brown skittles at:")
    for x, y in brown_coords:
        print(f"  - ({x}, {y})")

    return len(green_coords), len(brown_coords), green_coords, brown_coords


# Overlay centroid markers for green and brown skittles on the image.
def highlight(image, green_coords, brown_coords, ax=None):

    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(image)
    for x, y in green_coords:
        ax.plot(x, y, 'ro', markersize=4)  # Red dots for green skittles
    for x, y in brown_coords:
        ax.plot(x, y, 'wo', markersize=4)  # White dots for brown skittles



if __name__ == "__main__":
    # Load input image
    source = io.imread('C:/Users/alexm/OneDrive/Documents/Engineering Skills/Python/20230113_102538.jpg')

    # Define color thresholds (easy to adjust)
    green_ref = {"center": [35, 95, 20], "radius": 60}
    brown_ref = {"center": [74, 4, 4], "radius": 65}

    # Process image
    num_greens, num_browns, green_coords, brown_coords = process(source, green_ref, brown_ref)

    # Visualize results
    highlight(source, green_coords, brown_coords)
    plt.show()
