import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import time as t
import random
import math

class DMatch:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def find_feature(img):
    print('start looking for features...')
    sift = cv2.xfeatures2d.SIFT_create()
    kp,des = sift.detectAndCompute(img,None)
    
    return kp, des

def feature_match(des1,des2):
    print('start matching descriptors...')
    matches = []
    n = des1.shape[0]
    for i in range(n):
        ssd = np.sum((des2 - des1[i,:,])**2,axis=1)**(1./2)
        id_1, id_2 = np.argsort(ssd)[:2]
        ratio = ssd[id_1]/ssd[id_2]
        if ratio <= 0.8:
            data = {
                "distance" : ssd[id_1],
                "trainIdx" : id_1,
                "queryIdx" : i,
                "imgIdx" : 0
            }
            matches.append(DMatch(**data))
    
    matches = sorted(matches, key=lambda x: x.distance)
    num_good_matches = int(len(matches) * 0.25)
    matches = matches[:num_good_matches]
    
    return matches

def ransacShift(corr, inline_dist, inline_threshold, time):
    finalshift = []
    max_inline_count = -1
    corr_count = len(corr)
    for i in range(time):
        match_p = corr[random.randrange(0, corr_count)]
        shift = match_p[0] - match_p[1]
        shifted = corr[:, 1] + shift
        diff = corr[:, 0] - shifted

        error = np.sqrt((diff**2).sum(axis = 1))
        inline_count = len(np.where(error < inline_dist))

        if inline_count > max_inline_count:
            finalshift = shift
            max_inline_count = inline_count
        if max_inline_count > (corr_count * inline_threshold):
            break
    return np.rint(finalshift).astype(int)

def cylinderProject(img, f):
    x_length = img.shape[1]
    y_length = img.shape[0]
    cylinderImg = np.zeros(shape = img.shape, dtype =np.uint8)

    for idx_y in range(int(-y_length/2), int(y_length/2)):
        for idx_x in range(int(-x_length/2), int(x_length/2)):
            cylinederIdx_x = f * math.atan(idx_x / f)
            cylinederIdx_y = f * idx_y / math.sqrt(idx_x**2 + f**2)

            cylinederIdx_x = round(cylinederIdx_x + x_length/2)
            cylinederIdx_y = round(cylinederIdx_y + y_length/2)

            if cylinederIdx_x >= 0 and cylinederIdx_x < x_length and cylinederIdx_y >= 0 and cylinederIdx_y < y_length:
                    cylinderImg[cylinederIdx_y][cylinederIdx_x] = img[int(idx_y + y_length/2)][int(idx_x + x_length/2)]
    gray = cv2.cvtColor(cylinderImg, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    return cylinderImg[y:y+h, x:x+w]

def stitch(left_img, right_img):
    left_gray= cv2.cvtColor(left_img,cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img,cv2.COLOR_BGR2GRAY)
        
    kp1, des1 = find_feature(left_gray)
    kp2, des2 = find_feature(right_gray)
        
    matches = feature_match(des1,des2)
    #matches = cv2_feature_match(des1, des2)

    corrList = []
    for i in matches:
        lx,ly = kp1[i.queryIdx].pt
        rx,ry = kp2[i.trainIdx].pt
        pair = np.asarray([[lx, ly], [rx, ry]])
        corrList.append(pair)
    corrList = np.asarray(corrList)
        # corrList.append([lx, ly, rx, ry])
    # Do Ransac and Calculate global shift
    distance = 3
    threshold = 0.9
    time = 1000
    shift = ransacShift(corrList, distance, threshold, time)
    print("shift")
    print(shift)
    return shift

def blend(l_img, r_img, shift):
    padding = [(shift[1], 0) if shift[1] > 0 else (0, -shift[1]), 
            (shift[0], 0) if shift[0] > 0 else (0, -shift[0]),
            (0, 0)]
    shiftR = np.pad(r_img, padding, 'constant', constant_values=0)
    blend_shape = l_img.shape[1] + abs(shift[0])
    concated_part = shiftR[:, blend_shape:]
    shiftR = shiftR[:, :blend_shape]
    midle = blend_shape//2
    half_window = 3
    left_initail_y = 0 if shift[0] > 0 else -shift[0]
    shiftR[left_initail_y:(l_img.shape[0] + left_initail_y), 0:(midle-half_window)] = l_img[0:l_img.shape[0], 0:(midle-half_window)]
    for i in range(half_window*2):
        ratio = i / (half_window*2)
        shiftR[left_initail_y:(l_img.shape[0] + left_initail_y), (midle-half_window) + i] = \
            (1-ratio) * l_img[:, (midle-half_window) + i] + \
            ratio * shiftR[left_initail_y:(l_img.shape[0] + left_initail_y), (midle-half_window) + i]
    shiftR = np.concatenate((shiftR, concated_part), axis = 1)
    return shiftR

def parse(data_path, file_name):
    imgs = []
    focals = []
    f = open(file_name)
    for line in f:
        (img, focal, *others) = line.split()
        imgs.append(img)
        focals.append(float(focal))
    imglist = [cv2.imread("./"+ data_path + "/"+ file) for file in imgs]
    return(imglist, focals)

if __name__ == '__main__':
    data_path = 'parrington'
    file_name = 'pano_list.txt'
    imglist, focals = parse(data_path, file_name)
    cylinderImgs = [cylinderProject(imglist[i], focals[i]) for i in range(len(imglist))]
    xshape = 0
    
    shifts = []
    ConcateImg = cylinderImgs[0].copy()
    for i in range(1, len(cylinderImgs)):

        r_image = cylinderImgs[i-1]
        xshape += r_image.shape[1]

        l_image = cylinderImgs[i]
        shifts.append(stitch(l_image, r_image))
        ConcateImg = blend(l_image, ConcateImg, shifts[-1])
        print("==========================================================")
    cv2.imwrite('output.jpg', ConcateImg)