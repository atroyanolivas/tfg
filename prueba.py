from PIL import Image
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

list_dir = os.listdir(path := "./dataset/1/")

# Medir por porcentaje de escala en vez de hard code
# LEFT = 0.20
LEFT = 0.20
UPPER = 0.40
RIGHT = 0.40
LOWER = 0.60

LEFT_l = 0.63
UPPER_l = 0.23
RIGHT_l = 0.83
LOWER_l = 0.43


left = LEFT
upper = UPPER
right = RIGHT
lower = LOWER

print(f"Opening {list_dir[1]} in directory {path}")

# def get_image_crop(image: Image, centre: tuple):
#     """
#     centre: in percentajes (y, x) (y, x being the axes)
#     """
#     crop = image.crop((
#         image.size[0] * centre[1] - 32,
#         image.size[1] * centre[0] - 32,
#         image.size[0] * centre[1] + 32,
#         image.size[1] * centre[0] + 32
#     ))

#     return crop

# def get_rectangle(image, centre):
#     rect = patches.Rectangle(
#         (image.size[0] * centre[1] - 32, image.size[1] * centre[0] - 32),
#         64,
#         64,
#         linewidth=2, edgecolor='r', fill=False
#     )

#     return rect

def divide_image(image, rows=3, cols=3):

    img_width, img_height = image.size
    subimg_width = img_width // cols
    subimg_height = img_height // rows

    subimages = []
    for y in range(rows):
        for x in range(cols):
            left = x * subimg_width
            upper = y * subimg_height
            right = left + subimg_width
            lower = upper + subimg_height
            subimage = image.crop((left, upper, right, lower))
            subimages.append(subimage)

    return subimages

def get_2_crops(image: Image, factor=1.5) -> tuple:
    cropped_image_left = image.crop((image.size[0] * LEFT,
                                    image.size[1] * UPPER, image.size[0] * RIGHT,
                                    image.size[1] * LOWER))
    cropped_image_right = image.crop((image.size[0] * LEFT_l,
                                    image.size[1] * UPPER_l, image.size[0] * RIGHT_l,
                                    image.size[1] * LOWER_l))
    return cropped_image_left, cropped_image_right

# Load the PIL image
for im in list_dir:
    fig, axs = plt.subplots(1, 3)
    pil_image = Image.open(path + im)
    print(pil_image.size)
    # cropped_image = get_image_crop(pil_image, centre := (0.2, 0.4))
    # rectangle = get_rectangle(pil_image, (0.2, 0.4))

    cropped_image_left, cropped_image_right = get_2_crops(pil_image)

    rectangle_left = patches.Rectangle((pil_image.size[0] * left, pil_image.size[1] * lower),
                                       pil_image.size[0] * (right-left), 
                                       pil_image.size[1] * (upper-lower), linewidth=2, edgecolor='r', fill=False)
    rectangle_right = patches.Rectangle((pil_image.size[0] * LEFT_l, pil_image.size[1] * LOWER_l),
                                        pil_image.size[0] * (RIGHT_l-LEFT_l),
                                        pil_image.size[1] * (UPPER_l-LOWER_l), linewidth=2, edgecolor='r', fill=False)
    
    # clahe = cv2.createCLAHE(clipLimit=0.7, tileGridSize=(1, 1))

    # Add the rectangle patch to the axis
    axs[0].imshow(im_array := np.array(pil_image))
    axs[0].add_patch(rectangle_left)
    axs[0].add_patch(rectangle_right)
    # axs[1, 0].imshow(np.array(
    #     cv2.equalizeHist(
    #         cv2.imread(path + im, cv2.IMREAD_GRAYSCALE)
    #     )
    # ))
    # axs[1, 0].add_patch(rectangle_left)
    # axs[1, 0].add_patch(rectangle_right)
    # im_array = np.uint8(cv2.normalize(im_array, None, 0, 255, cv2.NORM_MINMAX))
    # axs[1, 0].imshow(clahe.apply(im_array))

    axs[1].imshow(arr_left := np.array(cropped_image_left))  # To display the cropped image
    axs[2].imshow(arr_left := np.array(cropped_image_right))

    subimg_width = cropped_image_left.size[0] // (cropped_image_left.size[0] // 64)
    # subimg_height = img_height // rows

    for i in range(cropped_image_left.size[0] // 64):
        axs[1].axvline(x=i * subimg_width, color='r', linewidth=2)
        axs[1].axhline(y=i * subimg_width, color='r', linewidth=2)

    print("Cropped image size:", cropped_image_left.size)
    
    im_list = divide_image(cropped_image_left, rows=cropped_image_left.size[0] // 64, cols=cropped_image_left.size[1] // 64)

    print("Number of subimages:", len(im_list))
    # for i in range(0, len(im_list)):
    #     axs[1, i].imshow(np.array(im_list[i]))
    plt.show()
