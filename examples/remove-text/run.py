import base64
import matplotlib.pyplot as plt
import keras_ocr
import cv2
import math
import numpy as np


def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2) / 2)
    y_mid = int((y1 + y2) / 2)
    return (x_mid, y_mid)


def remove_text(**inputs):
    # Grab the base64 from the kwargs
    encoded_image = inputs["image"]
    # Convert the base64-encoded input image to a buffer
    image_buffer = base64.b64decode(encoded_image)

    pipeline = keras_ocr.pipeline.Pipeline()

    # Read the image
    img = keras_ocr.tools.read(image_buffer)
    # Generate (word, box) tuples
    prediction_groups = pipeline.recognize([img])
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1]
        x2, y2 = box[1][2]
        x3, y3 = box[1][3]

        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)

        thickness = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255, thickness)
        img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Save the generated image to the Beam Output path
    cv2.imwrite("output.png", img_rgb)


if __name__ == "__main__":
    input_image = "./coffee.jpeg"
    with open(input_image, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read())
        remove_text(image=encoded_image)
