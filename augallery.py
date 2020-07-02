from glob import glob

import cv2
import numpy as np
import streamlit as st

from augtypes import rgb_shift_option, hsv_shift_option, blur_option

DATA_PATH = './samples/'
IMAGE_SIZE = (500 // 3) * 3
IMAGE_PATH = data_path_list = glob(DATA_PATH + '*.png')[0]

def main():
    st.write("# Welcome to Augallery!")

    option = st.sidebar.selectbox("Augmentation type", options=["Choose an augtype", "RGB shift", "HSV shift", "Blur"])

    if option == "RGB shift":
        rgb_shift_option(IMAGE_PATH, IMAGE_SIZE)
    elif option == "HSV shift":
        hsv_shift_option(IMAGE_PATH, IMAGE_SIZE)
    elif option == "Blur":
        blur_option(IMAGE_PATH, IMAGE_SIZE)


if __name__ == '__main__':
    main()