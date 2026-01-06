import pytest
from cyo.api.image_processing import encryption, decryption
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np


@pytest.mark.parametrize("image_path, key, left, right, top, bottom", [
    ("static/000018.jpg", '0.1,0.1', 148, 349, 85, 359)
])
def test_image_encrypt_and_decrypt(image_path, key, left, right, top, bottom):

    # image = Image.open(image_path)
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    image_en = image
    image_en[top:bottom, left:right, :] = encryption(image[top:bottom, left:right, :], key)
    image_en = Image.fromarray(image_en)
    plt.imshow(image_en)
    plt.show()

    image_en = np.array(image_en)
    image_de = image_en
    image_de[top:bottom, left:right, :] = decryption(image_en[top:bottom, left:right, :], key)

    image_de = Image.fromarray(image_en)
    plt.imshow(image_de)
    plt.show()

