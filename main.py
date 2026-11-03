
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

# ---------------------------------
# 1. SELECT INPUT IMAGE
# ---------------------------------

root = Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Select an Image",
    filetypes=[
        ("Image Files", "*.jpg *.jpeg *.png")
    ]
)

if not image_path:
    print("No image selected!")
    exit()

# ---------------------------------
# 2. READ IMAGE
# ---------------------------------

image = cv2.imread(image_path)

if image is None:
    print("Unable to load image!")
    exit()

# Convert BGR to RGB for display
original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ---------------------------------
# 3. GRAYSCALE CONVERSION
# ---------------------------------

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ---------------------------------
# 4. ROBERTS EDGE DETECTION
# ---------------------------------

kernel_x = np.array([
    [1, 0],
    [0, -1]
], dtype=np.float32)

kernel_y = np.array([
    [0, 1],
    [-1, 0]
], dtype=np.float32)

gx = cv2.filter2D(gray, cv2.CV_32F, kernel_x)
gy = cv2.filter2D(gray, cv2.CV_32F, kernel_y)

roberts = cv2.convertScaleAbs(
    cv2.magnitude(gx, gy)
)

# ---------------------------------
# 5. PREWITT EDGE DETECTION
# ---------------------------------

prewitt_x = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=np.float32)

prewitt_y = np.array([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
], dtype=np.float32)

gx = cv2.filter2D(gray, cv2.CV_32F, prewitt_x)
gy = cv2.filter2D(gray, cv2.CV_32F, prewitt_y)

prewitt = cv2.convertScaleAbs(
    cv2.magnitude(gx, gy)
)

# ---------------------------------
# 6. SOBEL EDGE DETECTION
# ---------------------------------

sobel_x = cv2.Sobel(
    gray, cv2.CV_64F, 1, 0, ksize=3
)

sobel_y = cv2.Sobel(
    gray, cv2.CV_64F, 0, 1, ksize=3
)

sobel = cv2.convertScaleAbs(
    cv2.magnitude(sobel_x, sobel_y)
)

# ---------------------------------
# 7. LAPLACIAN EDGE DETECTION
# ---------------------------------

laplacian = cv2.Laplacian(
    gray, cv2.CV_64F
)

laplacian = cv2.convertScaleAbs(laplacian)

# ---------------------------------
# 8. CANNY EDGE DETECTION
# ---------------------------------

canny = cv2.Canny(gray, 100, 200)

# ---------------------------------
# 9. DISPLAY ALL RESULTS
# ---------------------------------

titles = [
    "Original Image",
    "Grayscale Image",
    "Roberts Operator",
    "Prewitt Operator",
    "Sobel Operator",
    "Laplacian Operator",
    "Canny Edge Detection"
]

images = [
    original,
    gray,
    roberts,
    prewitt,
    sobel,
    laplacian,
    canny
]

fig, axes = plt.subplots(
    2, 4, figsize=(18, 10)
)

axes = axes.flatten()

for i in range(len(images)):

    if i == 0:
        axes[i].imshow(images[i])
    else:
        axes[i].imshow(images[i], cmap="gray")

    axes[i].set_title(
        titles[i],
        fontsize=14,
        fontweight="bold",
        pad=15
    )

    axes[i].axis("off")

# Hide unused position
axes[7].axis("off")

fig.suptitle(
    "EDGE DETECTION AND IMAGE ANALYSIS TOOL",
    fontsize=19,
    fontweight="bold"
)

plt.tight_layout(rect=[0, 0, 1, 0.94])

# ---------------------------------
# 10. SAVE OUTPUT
# ---------------------------------

plt.savefig(
    "edge_detection_results.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\n--------------------------------")
print("EDGE DETECTION COMPLETED!")
print("--------------------------------")
print("Algorithms Applied:")
print("1. Roberts Operator")
print("2. Prewitt Operator")
print("3. Sobel Operator")
print("4. Laplacian Operator")
print("5. Canny Edge Detection")
print("\nOutput saved as edge_detection_results.png")
