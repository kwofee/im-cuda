import json

cells = []
code_blocks = [
    "import numpy as np\nimport matplotlib.pyplot as plt\nfrom PIL import Image\nimport my_extension",
    "# Load image and convert to grayscale\nimg = Image.open('sample.jpg').convert('L')\n# Convert to float32 numpy array\nimg_arr = np.array(img, dtype=np.float32)\nm, n = img_arr.shape\n\nplt.figure(figsize=(8, 8))\nplt.imshow(img_arr, cmap='gray')\nplt.title('Original Image (Grayscale)')\nplt.axis('off')\nplt.show()",
    "# Helper function to apply filter and display the result\ndef apply_and_show(filter_obj, title):\n    out = np.zeros((m, n), dtype=np.float32)\n    filter_obj.apply(img_arr, out, m, n)\n    \n    plt.figure(figsize=(8, 8))\n    # For derivative filters, we might have negative values, \n    # but matplotlib's imshow with cmap='gray' auto-scales min to max, \n    # making visualization easy.\n    plt.imshow(out, cmap='gray')\n    plt.title(title)\n    plt.axis('off')\n    plt.show()\n    return out",
    "sobel_x = my_extension.SobelFilter('x')\nout_sx = apply_and_show(sobel_x, 'Sobel X (Horizontal Edges)')",
    "sobel_y = my_extension.SobelFilter('y')\nout_sy = apply_and_show(sobel_y, 'Sobel Y (Vertical Edges)')",
    "scharr_x = my_extension.ScharrFilter('x')\nout_schx = apply_and_show(scharr_x, 'Scharr X')",
    "lap4 = my_extension.LaplacianFilter(4)\nout_l4 = apply_and_show(lap4, 'Laplacian (4-connected)')",
    "lap8 = my_extension.LaplacianFilter(8)\nout_l8 = apply_and_show(lap8, 'Laplacian (8-connected)')",
    "gauss3 = my_extension.GaussianBlur(3, 1.0)\nout_g3 = apply_and_show(gauss3, 'Gaussian Blur (3x3, sigma=1.0)')",
    "gauss7 = my_extension.GaussianBlur(7, 2.0)\nout_g7 = apply_and_show(gauss7, 'Gaussian Blur (7x7, sigma=2.0)')"
]

for code in code_blocks:
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.split("\n")]
    })

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('demo.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

