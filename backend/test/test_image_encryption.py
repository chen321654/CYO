import pytest
from cyo.api.image_processing import encryption
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv


@pytest.mark.parametrize("image_path, key, left, right, top, bottom", [
    ("static/000018.jpg", '0.1,0.1', 148, 349, 85, 359)
])
def test_image_encryption(image_path, key, left, right, top, bottom):

    # image = Image.open(image_path)
    image = cv.imread(image_path)
    # image_1 = image[top:bottom, left:right]
    image_en = image
    image_en[top:bottom, left:right, :] = encryption(image[top:bottom, left:right, :], key)
    image_en = Image.fromarray(image)
    plt.imshow(image_en)
    plt.show()
