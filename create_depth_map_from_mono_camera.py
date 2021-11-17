import cv2 as cv
from PIL import Image
from flatbuffers.builder import np
from matplotlib import cm
from get_depth_from_image import get_depth_from_single_image


def create_depth_map_from_mono_camera(number_of_images):
    image_counter = 0
    video_capture = cv.VideoCapture("test_img.png")
    while image_counter <= number_of_images:
        is_successful, image = video_capture.read()
        if is_successful:
            get_depth_from_single_image(Image.fromarray(np.uint8(image)).convert('RGB'))
            image_counter += 1
        else:
            continue
    cv.destroyAllWindows()
    video_capture.release()


create_depth_map_from_mono_camera(0)
