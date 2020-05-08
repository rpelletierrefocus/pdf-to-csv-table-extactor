# pip install minecart
import array
import csv
import math

import minecart
import cv2
import numpy as np
import pytesseract as pytesseract
from imutils.perspective import four_point_transform
from imutils.contours import sort_contours
import argparse
import PIL
import io
from PIL import ExifTags
from pdf2image import convert_from_path
import os
import re

DEBUG = lambda: None
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = './tesseract/tesseract'
    print(pytesseract.pytesseract.tesseract_cmd)
DEBUG.isDebug = False
ROW_MIN_HEIGHT = 5  # in px
COLUMN_MIN_WIDTH = 5
PADDING = 2

def setDebug(debug):
    DEBUG.isDebug = debug


def PDFTOImage(filePath, outPath):
    poppler_path = None
    if os.name == 'nt':
        poppler_path= './poppler/bin'
    page = convert_from_path(filePath,  poppler_path=poppler_path)
    page[0].save(outPath, 'JPEG')
    return outPath

def resizeImageIfRequired(imageFile):

    basewidth = 1969
    img = PIL.Image.open(imageFile)
    angleToRotate = 0
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        exif = dict(img._getexif().items())

        if exif[orientation] == 3:
            angleToRotate = 180
        elif exif[orientation] == 6:
            angleToRotate = 270            
        elif exif[orientation] == 8:
            angleToRotate = 90

    except Exception as e:
        print("Image meta not found: going to use pytesseract.image_to_osd:", e)
        try:
            angleToRotate=360-int(re.search('(?<=Rotate: )\d+', pytesseract.image_to_osd(img)).group(0))
        except Exception as err:
             print(err)
        
    img = img.rotate(angleToRotate, expand=True)
    return img

def parseCompleteText(im, outPath, config, postfix=""):
    try :   
        text = pytesseract.image_to_string(im, config=config)
        text_file = open(outPath + postfix+'.txt', 'w', encoding="utf-8")
        n = text_file.write(text)
        text_file.close()
    except Exception as e:
         print("err:", e)

def process_file(filename, outPath, improveBackground=False):
    config = ("-l ron --oem 1 --psm 6")

    im = resizeImageIfRequired(open(filename, 'rb'))
    if not improveBackground:
        parseCompleteText(im, outPath, config, "_non_blur")

    imageFile = open(filename, 'rb')
    with open(outPath + '.csv', 'w', encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        im = im.convert('L')  # validate grayscale
        if DEBUG.isDebug:
            im.show()
        gray_image = np.array(im)
        
        extracted_table = extract_main_table(gray_image, outPath, config)
        if DEBUG.isDebug:
            show_wait_destroy("extracted", extracted_table)
        row_images = extract_rows_columns(extracted_table)  # [1:]
        if len(row_images) == 0:
            return

        idx = 0
        for row in row_images:
            idx += 1
            print("%s : Extracting row %d out of %d file %s" %
                  (outPath, idx, len(row_images), filename))
            row_texts = []
            for column in row:
                try:
                    text = pytesseract.image_to_string(column, config=config)
                    row_texts.append(text)
                except Exception as e:
                    print("err:", e)
                    row_texts.append("")
                    continue

            csv_writer.writerow(row_texts)


def extract_main_table(gray_image, outPath, config):
    inverted = cv2.bitwise_not(gray_image)
    
    if DEBUG.isDebug:
        show_wait_destroy("inverted", inverted)

    blurred = cv2.GaussianBlur(inverted, (5, 5), 0)

    thresholded = cv2.threshold(blurred, 0, 255,
                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    if DEBUG.isDebug:
        show_wait_destroy("thresholded", thresholded)

    superInverted = cv2.bitwise_not(thresholded)
    parseCompleteText(superInverted, outPath, config)

    if DEBUG.isDebug:
        show_wait_destroy("superInverted", superInverted)
    cnts = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1]  # if imutils.is_cv2() else cnts[1]

    output = thresholded.copy()
    for c in cnts:
        # draw each contour on the output image with a 3px thick purple
        # outline, then display the output contours one at a time
        cv2.drawContours(output, [c], -1, (240, 0, 159), 3)

    if DEBUG.isDebug:    
        show_wait_destroy("Contours", output)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    rect = cv2.minAreaRect(cnts[0])
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    extracted = four_point_transform(gray_image.copy(), box.reshape(4, 2))

    if DEBUG.isDebug:
        color_image = cv2.cvtColor(gray_image.copy(), cv2.COLOR_GRAY2BGR)
        cv2.drawContours(color_image, [box], 0, (0, 0, 255), 2)
        cv2.drawContours(color_image, [cnts[0]], -1, (0, 255, 0), 2)
        
        show_wait_destroy("thresholded", color_image)
        show_wait_destroy("extracted", extracted)
    return extracted


def horizontal_boxes_filter(box, width):
    x, y, w, h = box
    return w > width * 0.1


def vertical_boxes_filter(box, height):
    x, y, w, h = box
    return h > height * 0.1


def extract_rows_columns(gray_image):
    inverted = cv2.bitwise_not(gray_image)
    blurred = cv2.GaussianBlur(inverted, (5, 5), 0)

    height, width = gray_image.shape

    thresholded = cv2.threshold(blurred, 128, 255,
                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    if DEBUG.isDebug:
        show_wait_destroy("extract_rows_columns", thresholded)
    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    vertical_kernel_height = math.ceil(height*0.1)
    verticle_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, vertical_kernel_height))

    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    horizontal_kernel_width = math.ceil(width*0.1)
    hori_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (horizontal_kernel_width, 1))

    # Morphological operation to detect vertical lines from an image
    img_temp1 = cv2.erode(thresholded, verticle_kernel, iterations=3)
    if DEBUG.isDebug:
        show_wait_destroy("extract_rows_columns", img_temp1)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
    if DEBUG.isDebug:
        show_wait_destroy("extract_rows_columns", verticle_lines_img)
    _, vertical_contours, _ = cv2.findContours(verticle_lines_img.copy(), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
    # Sort all the contours by top to bottom.
    (vertical_contours, vertical_bounding_boxes) = sort_contours(
        vertical_contours, method="left-to-right")

    filtered_vertical_bounding_boxes = list(
        filter(lambda x: vertical_boxes_filter(x, height), vertical_bounding_boxes))

    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(thresholded, hori_kernel, iterations=3)

    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
    if DEBUG.isDebug:
        show_wait_destroy("extract_rows_columns", horizontal_lines_img)
    _, horizontal_contours, _ = cv2.findContours(horizontal_lines_img.copy(), cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_SIMPLE)

    horizontal_contours, horizontal_bounding_boxes = sort_contours(
        horizontal_contours, method="top-to-bottom")

    filtered_horizontal_bounding_boxes = list(
        filter(lambda x: horizontal_boxes_filter(x, width), horizontal_bounding_boxes))

    # if DEBUG.isDebug:
    color_image = cv2.cvtColor(gray_image.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(color_image, vertical_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(color_image, horizontal_contours, -1, (255, 0, 0), 2)

    # for filtered_horizontal_bounding_box in filtered_horizontal_bounding_boxes:
    #     x,y,w,h = filtered_horizontal_bounding_box
    #     cv2.rectangle(color_image,(x,y),(x+w,y+h),(0,255,255),2)
    #
    # for filtered_vertical_bounding_box in filtered_vertical_bounding_boxes:
    #     x,y,w,h = filtered_vertical_bounding_box
    #     cv2.rectangle(color_image,(x,y),(x+w,y+h),(0,255,255),2)
    if DEBUG.isDebug:
        show_wait_destroy("horizontal_vertical_contours", color_image)

    extracted_rows_columns = []

    for idx_h, horizontal_bounding_box in enumerate(filtered_horizontal_bounding_boxes):
        if idx_h == 0:
            continue
        # previous horizontal box
        hx_p, hy_p, hw_p, hh_p = filtered_horizontal_bounding_boxes[idx_h-1]
        hx_c, hy_c, hw_c, hh_c = horizontal_bounding_box

        extracted_columns = []
        for idx_v, vertical_bounding_box in enumerate(filtered_vertical_bounding_boxes):
            if idx_v == 0:
                continue
            # previous horizontal box
            vx_p, vy_p, vw_p, vh_p = filtered_vertical_bounding_boxes[idx_v-1]
            vx_c, vy_c, vw_c, vh_c = vertical_bounding_box
            table_cell = gray_image[hy_p:hy_c+hh_c, vx_p:vx_c+vw_c]

            blurred = cv2.GaussianBlur(table_cell, (5, 5), 0)
            # cv2.rectangle(color_image,(vx_p,hy_p),(vx_c+vw_c,hy_c+hh_c),(255,0,0),2)

            thresholded = cv2.threshold(blurred, 128, 255,
                                        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            im2, contours, hierarchy = cv2.findContours(
                thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            rect = cv2.minAreaRect(contours[0])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            extracted = four_point_transform(table_cell.copy(), box.reshape(4, 2))[
                1:-1, 1:-1]  # remove 1 px from each side
            ret, extracted = cv2.threshold(
                extracted, 165, 255, cv2.THRESH_BINARY)
            extracted_columns.append(extracted)

            cv2.drawContours(color_image, [contours[0]], -1, (0, 255, 0), 3)

        extracted_rows_columns.append(extracted_columns)
    if DEBUG.isDebug:
        show_wait_destroy("horizontal_lines_img", color_image)
    return extracted_rows_columns


def show_wait_destroy(winname, img):
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.imshow(winname, img)
    cv2.resizeWindow(winname, 500, 500)
    cv2.moveWindow(winname, 500, 0)
    cv2.waitKey(0)
    cv2.destroyWindow(winname)


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--pdf", type=str, nargs='+', required=True,
                    help="path to input pdf of scans (accepts multiple pdf files too)")

    args = vars(ap.parse_args())
    for file_name in args["pdf"]:
        process_file(file_name)
