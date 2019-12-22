import cv2
import numpy as np
from matplotlib import pyplot as plt

def process_picture(image_path):
    image_bgr = cv2.imread(image_path)
    image_bgr = cv2.resize(image_bgr, (256, 256))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rectangle = (50, 0, 200, 250)

    # Create initial mask
    mask = np.zeros(image_rgb.shape[:2], np.uint8)

    # Create temporary arrays used by grabCut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Run grabCut
    cv2.grabCut(image_rgb, # Our image
                mask, # The Mask
                rectangle, # Our rectangle
                bgdModel, # Temporary array for background
                fgdModel, # Temporary array for background
                5, # Number of iterations
                cv2.GC_INIT_WITH_RECT) # Initiative using our rectangle

    # Create mask where sure and likely backgrounds set to 0, otherwise 1
    mask_2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')
    plt.imshow(mask_2)
    plt.show()

    # Multiply image with new mask to subtract background
    image_rgb_nobg = image_rgb * mask_2[:, :, np.newaxis]

    plt.imshow(image_rgb_nobg), plt.axis("off")
    plt.show()

    plt.imshow(image_bgr), plt.axis("off")
    plt.show()
    cv2.imwrite('./test.png', image_rgb_nobg)

if __name__ == "__main__":
    for i in range(1, 15):
        process_picture('./yolo-train/data/{}.jpg'.format(i))
