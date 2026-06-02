from PIL import Image
import numpy as np

# 4x4 binary image with duplicate 2x2 chunks:
#
# chunk 0 == chunk 3
#
# [[1,0 | 0,1],
#  [1,1 | 1,0],
#  -----+-----
#  [0,0 | 1,0],
#  [1,0 | 1,1]]

arr = np.array([
    [255,   0,   0, 255],
    [255, 255, 255,   0],
    [  0,   0, 255,   0],
    [255,   0, 255, 255],
], dtype=np.uint8)

Image.fromarray(arr, mode="L").save("all_pairs_similarity_analysis/test.png")