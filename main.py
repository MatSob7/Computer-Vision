import numpy as np
import cv2
import sys
import math

def do_bilateral_filter(img, filtered_image, x, y, sigma_i, sigma_s):
    diameter = sigma_s
    hl = np.round(diameter/2)
    i_fil = 0
    Wp = 0
    i = 0
    while i < diameter:
        j = 0
        while j < diameter:
            neigx = int(x - (hl - i))
            neigy = int(y - (hl - j))
            if neigx >= len(img):
                neigx -= len(img)
            if neigy >= len(img[0]):
                neigy -= len(img[0])
            gi = gauss(img[neigx][neigy] - img[x][y], sigma_i)
            gs = gauss(distance(neigx, neigy, x, y), sigma_s)
            w = gi * gs
            i_fil += img[neigx][neigy] * w
            Wp += w
            j += 1
        i += 1
    i_fil = i_fil / Wp
    filtered_image[x][y] = int(round(i_fil))


def bilateral_filter(source, sigma_i, sigma_s):
    filtered_image = np.zeros(source.shape)
    i = 0
    while i < len(source):
        j = 0
        while j < len(source[0]):
            do_bilateral_filter(source, filtered_image, i, j, sigma_i, sigma_s)
            j += 1
        i += 1
    return filtered_image


def distance(x, y, i, j):
    return np.sqrt((x-i)**2 + (y-j)**2)

def gauss(x, sigma):
    return (1.0 / (2 * math.pi * (sigma * 2))) * math.exp(- (x * 2) / (2 * sigma ** 2))


I = cv2.imread('pierozki.png')

B = np.stack([
        bilateral_filter( I[:,:,0], 10, 1 ),
        bilateral_filter( I[:,:,1], 10, 1 ),
        bilateral_filter( I[:,:,2], 10, 1 )], axis=2 )



cv2.imwrite( 'bilateral2.png',B )