import cv2
import numpy as np
from skimage.filters import threshold_multiotsu
import math
from math import fabs, sqrt
from numpy.core.numeric import False_
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity

import matplotlib.pyplot as plt

def two_level_threshold(image):
    thresholds = threshold_multiotsu(image)
    img_thres1 = cv2.threshold(image, thresholds[0], 255, cv2.THRESH_BINARY_INV)[1]
    #img_thres2 = cv2.threshold(image, thresholds[1], 255, cv2.THRESH_BINARY_INV)[1]
    #anded = cv2.bitwise_and(img_thres1, img_thres2)
    return img_thres1

def extract_contours(image):
    image = cv2.GaussianBlur(image, (5, 5), 0)
    kernel = np.ones((3, 3), np.uint8)

    img_thres_otsu = two_level_threshold(image)
    img_thres_otsu = cv2.erode(img_thres_otsu, kernel)
    img_thres_otsu = cv2.dilate(img_thres_otsu, kernel)

    contours, _ = cv2.findContours(img_thres_otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    largest_contour, largest_area = None, 0
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area > largest_area:
            largest_area = area
            largest_contour = cnt

    if largest_contour is None:
        return None, None

    box = cv2.boundingRect(largest_contour)
    contour_image = np.zeros((image.shape[0], image.shape[1], 1), np.uint8)
    cv2.drawContours(contour_image, [largest_contour], -1, 255, -1)
    contour_image = contour_image[box[1]: box[1] + box[3], box[0]: box[0] + box[2]]
    return contour_image, largest_contour

def compare_binary_images(image, image_ref, similarity_threshold=0.90):

    # intesection over union (IOU)
    img_size = image.shape
    image = cv2.resize(image, (img_size[1], img_size[0]))
    image_ref = cv2.resize(image_ref, (img_size[1], img_size[0]))
    anded = cv2.bitwise_and(image, image_ref)
    ored = cv2.bitwise_or(image, image_ref)

    # cv2.imshow("anded", anded)
    # cv2.imshow("ored", ored)

    anded_count = cv2.countNonZero(anded)
    ored_count = cv2.countNonZero(ored)

    # print("Anded:" + str(anded_count))
    # print("ored_count:" + str(ored_count))

    similarity_score = round(anded_count / ored_count, 2)

    return similarity_score > similarity_threshold, similarity_score

def compare_contours(cnt_c, cnt_f, similarity_threshold=0.90):
    acceptable_error = 1.5
    score = cv2.matchShapes(cnt_c, cnt_f, 1, 0.0)
    score_scaled = round(max(1.0 - score / (acceptable_error * 10), 0.0), 2)
    return score < score_scaled, score, score_scaled

def center_of_gravity(image):
    M = cv2.moments(image)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.circle(image, (cX, cY), 5, (160, 160, 160), -1)
    cv2.putText(image, "Centroid", (cX , cY ),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 2)
    
    return image

def retrection(image_true,image_error):  
    #image format RGB  
    (h,w,c)=image_true.shape
    image_error=cv2.resize(image_error,(w,h),interpolation=cv2.INTER_AREA)
    image_c=image_error-image_true

    kernel=np.ones((5,5),np.uint8)
    image_f=cv2.erode(image_c,kernel,iterations=1)
    image_f=cv2.cvtColor(image_f, cv2.COLOR_BGR2GRAY)
    kontur,_=cv2.findContours(image_f,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for k in kontur:
        (x, y, w, h) = cv2.boundingRect(k)
        cv2.rectangle(image_f, (x, y), (w+x, h+y), (255, 255, 255), 2)
    
    image_f=cv2.cvtColor(image_f,cv2.COLOR_GRAY2BGR)
    image_f=cv2.addWeighted(image_f,0.6,image_error,0.4,1) 
    return image_f

def edgeLift(img):
    arrayTrue  = np.arange(8).reshape(2,4)
    arrayFalse = np.arange(8).reshape(2,4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    counter = 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 25, 0.3, 10)
    corners = np.int0(corners)
    deneme_x=[]
    deneme_y=[]
    for i in corners:
        counter = counter+1 
        x, y = i.ravel()
        deneme_x.append(x)
        deneme_y.append(y) 

        arrayFalse[0,counter-1] = x
        arrayFalse[1,counter-1] = y
        
        cv2.circle(img, (x,y), 3, 255, -1)
        cv2.putText(img,str(counter),(x+1,y+2),font,0.5,(255,255,255),1,cv2.LINE_AA)

        print("Point ID",counter, "Coordinat: x,y" , x,y)

    max_x=max(deneme_x)
    max_y=max(deneme_y)
    min_x=min(deneme_x)
    min_y=min(deneme_y)
    cv2.line(img,(min_x,max_y),(max_x,max_y),(255,123,85),2,1)#down
    cv2.line(img,(min_x,max_y),(max_x,min_y),(255,123,85),2,1)#up

    mesafe_1=abs((min_x-max_x)^2-(max_y-max_y)^2)
    mesafe_2=abs((min_x-max_x)^2-(max_y-min_y)^2)
    mesafe_3=abs((max_x-max_x)^2-(max_y-min_y)^2)

    distance_a=round(math.sqrt(mesafe_1))
    distance_b=round(math.sqrt(mesafe_2))
    distance_c=round(math.sqrt(mesafe_3))

    acı=float((distance_c^2-distance_a^2-distance_b^2)/2*distance_b*distance_c)

    angle=round(math.atan(acı),3)
    cv2.circle(img,(min_x,max_y),(80),(3,2,29),3)
    cv2.putText(img,f"{angle} degree",(min_x+80,max_y-8),2,1,(120,54,26))

    arrayTruex = np.arange(4).reshape(1, 4)
    arrayFalsex = np.arange(4).reshape(1, 4)

    arrayMin = np.arange(4).reshape(1, 4)
    arrayFalsey = np.arange(4).reshape(1, 4)

    difference = np.arange(16).reshape(4, 4)
    for i in range(0,4):
        arrayTruex[0][i] = arrayTrue[0][i]+arrayTrue[1][i]

    for i in range(0,4):
        arrayFalsex[0][i] = arrayFalse[0][i]+arrayFalse[1][i]

    #print("True : " , arrayTruex)
    #print("False : ", arrayFalsex)

    for i in range(0,4):
        arrayMin[0][i] = abs(arrayTruex[0][0]) - abs(arrayFalsex[0][i])

    min_result = abs(arrayMin).min() 
    #print(min_result)
    for j in range(0,4):
        if(abs(arrayMin[0][j]) == min_result):

            tempx = arrayFalse[0][j]
            tempy = arrayFalse[1][j]

            temp1x = arrayFalse[0][0]
            temp1y = arrayFalse[1][0]

            arrayFalse[0][0] = tempx
            arrayFalse[1][0] = tempy
            arrayFalse[0][j] = temp1x
            arrayFalse[1][j] = temp1y
    
    return img

def unSupport(image):
    #image format rgb
    image_gray = rgb2gray(image)
    blobs_doh = blob_doh(image_gray, max_sigma=7, threshold=.021)
    blobs_list = [blobs_doh]
    colors = ['red']
    titles = ['Support is Missing']
    sequence = zip(blobs_list, colors, titles)

    fig, axes = plt.subplots(2,1, figsize=(5, 2), sharex=False, sharey=True)
    ax = axes.ravel()

    for idx, (blobs, color, title) in enumerate(sequence):
        ax[idx].set_title(title)
        ax[idx].imshow(image)
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
            ax[idx].add_patch(c)

    plt.tight_layout()
    plt.show()
def score_show(image,score):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, f"Similarity Score: {round(score,2)}", (image.shape[0],30), font, 1, (0,0,255),2)

def diff(original,template):

    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    original_blur = cv2.GaussianBlur(original_gray, (5, 5), 3)
    template_blur = cv2.GaussianBlur(template_gray, (5, 5), 3)

    (score, diff) = structural_similarity(original_blur, template_blur, full=True)
    print("Image similarity", score)

    diff = (diff * 255).astype("uint8")

    _, thresh = cv2.threshold(
        diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    img_thres_otsu = cv2.morphologyEx(
        thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=lambda c: cv2.contourArea(c))

    mask = np.full_like(original, (190, 190, 190), dtype='uint8')
    filled_template = template.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    area = cv2.contourArea(contour)
    if area > 200:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(original, (x, y), (x + w, y + h), (36, 255, 12), 2)
        cv2.putText(original, "Correct Image",
                    (x-10, y-10), font, 1, (0, 0, 255), 2)

        cv2.rectangle(template, (x, y), (x + w, y + h), (36, 255, 12), 2)
        cv2.putText(template, "Failrue Image",
                    (x-10, y-10), font, 1, (0, 0, 255), 2)

        cv2.drawContours(mask, [contour], 0, (0, 255, 0), -1)
        cv2.putText(mask, "Contour Difference",
                    (x-10, y-10), font, 1, (0, 0, 255), 2)

        cv2.drawContours(filled_template, [contour], 0, (0, 255, 0), -1)
        cv2.putText(filled_template, "Filled Template",
                    (x-10, y-10), font, 1, (0, 0, 255), 2)

    stack1 = np.hstack((original, template))
    stack2 = np.hstack((mask, filled_template))
    stack = np.vstack((stack1, stack2))
    cv2.putText(stack, f"Similarity Score: {round(score,2)}",
                (stack.shape[0], 30), font, 1, (0, 0, 255), 2)

    cv2.imwrite("stack.jpg", stack)
    cv2.imshow('Stack', stack)
    cv2.waitKey(0)
