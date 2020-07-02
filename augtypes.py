from albumentations import Compose
from albumentations.augmentations.transforms import GaussianBlur, MedianBlur, MotionBlur
import cv2
import numpy as np
from PIL import Image
import streamlit as st

def blur_option(IMAGE_PATH, IMAGE_SIZE):
    img_bgr = cv2.imread(IMAGE_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (IMAGE_SIZE // 3, IMAGE_SIZE // 3))

    st.write("## Blur")
    op_list = [('gaussian', 'Gaussian Blur', GaussianBlur), 
               ('motion', 'Motion Blur', MotionBlur), 
               ('median', 'Median Blur', MedianBlur), ('combined', 'na', 'na')]

    blur_val_dict = {}
    blur_val_arr_dict = {}
    bg_dict = {}

    for i, (op, title, func) in enumerate(op_list):
        if title != 'na':
            blur_val_dict[op] = st.sidebar.slider(title, min_value=3, max_value=13, step=2, value=(3, 3))
            blur_val_arr_dict[op] = np.linspace(blur_val_dict[op][0], blur_val_dict[op][1], 9)
            blur_val_arr_dict[op] = [int(x) if int(x) % 2 == 1 else int(x) + 1 for x in blur_val_arr_dict[op]]

            print(op)
            print(blur_val_arr_dict[op])


        bg_dict[op] = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3)).astype(np.uint8)

        if op != 'combined':
            for j, blur_val in enumerate(blur_val_arr_dict[op]):
                # print("blur_val")
                # print(blur_val)
                func_inst = func(blur_limit=(int(blur_val), int(blur_val)), always_apply=True)

                row, col = divmod(j, 3)

                bg_dict[op][row * IMAGE_SIZE // 3 : row * IMAGE_SIZE // 3 + IMAGE_SIZE // 3, 
                           col * IMAGE_SIZE // 3 : col * IMAGE_SIZE // 3 + IMAGE_SIZE // 3, :] \
                           = func_inst(image=img_rgb)['image']
        else:
            for j in range(9):
                func_inst = Compose([GaussianBlur(blur_limit=(int(blur_val_arr_dict['gaussian'][j]), int(blur_val_arr_dict['gaussian'][j])), always_apply=True), 
                                     MedianBlur(blur_limit=(int(blur_val_arr_dict['median'][j]), int(blur_val_arr_dict['median'][j])), always_apply=True), 
                                     MotionBlur(blur_limit=(int(blur_val_arr_dict['motion'][j]), int(blur_val_arr_dict['motion'][j])), always_apply=True)
                                     ])

                row, col = divmod(j, 3)
                bg_dict[op][row * IMAGE_SIZE // 3 : row * IMAGE_SIZE // 3 + IMAGE_SIZE // 3, 
                           col * IMAGE_SIZE // 3 : col * IMAGE_SIZE // 3 + IMAGE_SIZE // 3, :] \
                           = func_inst(image=img_rgb)['image']
             
    total_background = np.zeros((IMAGE_SIZE * 2, IMAGE_SIZE * 2, 3)).astype(np.uint8)

    for i, effect in enumerate(('gaussian', 'motion', 'median', 'combined')):
        row, col = divmod(i, 2)
        total_background[row * IMAGE_SIZE : row * IMAGE_SIZE + IMAGE_SIZE,
                         col * IMAGE_SIZE : col * IMAGE_SIZE + IMAGE_SIZE,
                         :] = bg_dict[effect]

    st.image(total_background)

    st.markdown("* * * ")
    st.subheader("Parameters")
    st.write(f"Gaussian param: {blur_val_dict['gaussian']}")
    st.write(f"Motion param: {blur_val_dict['motion']}")
    st.write(f"Median: {blur_val_dict['median']}")

def rgb_shift_option(IMAGE_PATH, IMAGE_SIZE):
    img_bgr = cv2.imread(IMAGE_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (IMAGE_SIZE // 3, IMAGE_SIZE // 3))

    st.write("## RGB Shift")
    op_list = [('red', 'Red Shift', 'r_shift'), 
               ('green', 'Green Shift', 'g_shift'), 
               ('blue', 'Blue Shift', 'b_shift'), ('combined', 'na', 'na')]

    sft_dict = {}
    sft_arr_dict = {}
    bg_dict = {}

    for i, (op, title, arg) in enumerate(op_list):
        if title != 'na':
            sft_dict[op] = st.sidebar.slider(title, min_value=-255, max_value=255, value=(0, 0))
            sft_arr_dict[op] = np.linspace(sft_dict[op][0], sft_dict[op][1], 9)

        bg_dict[op] = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3)).astype(np.uint8)

        if op != 'combined':
            for j, shift in enumerate(sft_arr_dict[op]):
                row, col = divmod(j, 3)
                bg_dict[op][row * IMAGE_SIZE // 3 : row * IMAGE_SIZE // 3 + IMAGE_SIZE // 3, 
                           col * IMAGE_SIZE // 3 : col * IMAGE_SIZE // 3 + IMAGE_SIZE // 3, :] \
                           = rgb_shift(img_rgb, **{arg: shift})
        else:
            for j in range(9):
                row, col = divmod(j, 3)

                bg_dict[op][row * IMAGE_SIZE // 3 : row * IMAGE_SIZE // 3 + IMAGE_SIZE // 3, 
                           col * IMAGE_SIZE // 3 : col * IMAGE_SIZE // 3 + IMAGE_SIZE // 3, :] \
                           = rgb_shift(img_rgb, sft_arr_dict['red'][j], sft_arr_dict['green'][j], sft_arr_dict['blue'][j])
             
    total_background = np.zeros((IMAGE_SIZE * 2, IMAGE_SIZE * 2, 3)).astype(np.uint8)

    for i, color in enumerate(('red', 'green', 'blue', 'combined')):
        row, col = divmod(i, 2)
        total_background[row * IMAGE_SIZE : row * IMAGE_SIZE + IMAGE_SIZE,
                         col * IMAGE_SIZE : col * IMAGE_SIZE + IMAGE_SIZE,
                         :] = bg_dict[color]

    st.image(total_background)

    st.markdown("* * * ")
    st.subheader("Parameters")
    st.write(f"Red channel: {sft_dict['red']}")
    st.write(f"Green channel: {sft_dict['green']}")
    st.write(f"Blue channel: {sft_dict['blue']}")


def hsv_shift_option(IMAGE_PATH, IMAGE_SIZE):
    img_bgr = cv2.imread(IMAGE_PATH)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_hsv = cv2.resize(img_hsv, (IMAGE_SIZE // 3, IMAGE_SIZE // 3))

    st.write("## HSV Shift")
    op_list = [('hue', 'Hue Shift', 'h_shift'), 
               ('sat', 'Saturation Shift', 's_shift'), 
               ('val', 'Value', 'v_shift'), ('combined', 'na', 'na')]

    sft_dict = {}
    sft_arr_dict = {}
    bg_dict = {}

    for i, (op, title, arg) in enumerate(op_list):
        if title != 'na':
            if op != 'hue':
                sft_dict[op] = st.sidebar.slider(title, min_value=-255, max_value=255, value=(0, 0))
            else:
                sft_dict[op] = st.sidebar.slider(title, min_value=-180, max_value=180, value=(0, 0))
            sft_arr_dict[op] = np.linspace(sft_dict[op][0], sft_dict[op][1], 9)

        bg_dict[op] = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3)).astype(np.uint8)

        if op != 'combined':
            for j, shift in enumerate(sft_arr_dict[op]):
                row, col = divmod(j, 3)
                bg_dict[op][row * IMAGE_SIZE // 3 : row * IMAGE_SIZE // 3 + IMAGE_SIZE // 3, 
                           col * IMAGE_SIZE // 3 : col * IMAGE_SIZE // 3 + IMAGE_SIZE // 3, :] \
                           = hsv_shift(img_hsv, **{arg: shift})
        else:
            for j in range(9):
                row, col = divmod(j, 3)

                bg_dict[op][row * IMAGE_SIZE // 3 : row * IMAGE_SIZE // 3 + IMAGE_SIZE // 3, 
                           col * IMAGE_SIZE // 3 : col * IMAGE_SIZE // 3 + IMAGE_SIZE // 3, :] \
                           = hsv_shift(img_hsv, sft_arr_dict['hue'][j], sft_arr_dict['sat'][j], sft_arr_dict['val'][j])
             
    total_background = np.zeros((IMAGE_SIZE * 2, IMAGE_SIZE * 2, 3)).astype(np.uint8)

    for i, color in enumerate(('hue', 'sat', 'val', 'combined')):
        row, col = divmod(i, 2)
        total_background[row * IMAGE_SIZE : row * IMAGE_SIZE + IMAGE_SIZE,
                         col * IMAGE_SIZE : col * IMAGE_SIZE + IMAGE_SIZE,
                         :] = bg_dict[color]

    total_background = cv2.cvtColor(total_background, cv2.COLOR_HSV2RGB)
    st.image(total_background)

    st.markdown("* * * ")
    st.subheader("Parameters")
    st.write(f"Hue channel: {sft_dict['hue']}")
    st.write(f"Saturation channel: {sft_dict['sat']}")
    st.write(f"Value channel: {sft_dict['val']}")

def augment(aug, image):
    return aug(image=image)['image']


def rgb_shift(img, r_shift=0, g_shift=0, b_shift=0):
    img = img.copy().astype(float)

    shift_tuple = r_shift, g_shift, b_shift

    for i, shift in enumerate(shift_tuple):
        img[:, :, i] = np.where(img[:, :, i] + shift <= 255, img[:, :, i] + shift, 255)
        img[:, :, i] = np.where(img[:, :, i] >= 0, img[:, :, i], 0)

    return img.astype(np.uint8)


def hsv_shift(img, h_shift=0, s_shift=0, v_shift=0):
    img = img.copy().astype(float)

    shift_tuple = h_shift, s_shift, v_shift

    for i, shift in enumerate(shift_tuple):
        limit = 180 if i == 0 else 255

        img[:, :, i] = np.where(img[:, :, i] + shift <= limit, img[:, :, i] + shift, limit)
        img[:, :, i] = np.where(img[:, :, i] >= 0, img[:, :, i], 0)


    return img.astype(np.uint8)

    