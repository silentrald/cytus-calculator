import cv2
import numpy as np

img = cv2.imread('pictures/violet_drag.png', cv2.IMREAD_UNCHANGED)

def nothing(x):
    pass

cv2.namedWindow('Values')
cv2.createTrackbar('LH', 'Values', 0, 255, nothing)
cv2.createTrackbar('LS', 'Values', 0, 255, nothing)
cv2.createTrackbar('LV', 'Values', 0, 255, nothing)
cv2.createTrackbar('UH', 'Values', 0, 255, nothing)
cv2.createTrackbar('US', 'Values', 0, 255, nothing)
cv2.createTrackbar('UV', 'Values', 0, 255, nothing)

width = int(img.shape[1] / 2)
height = int(img.shape[0] / 2)
resized = cv2.resize(img, (width, height))
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

# cv2.imshow('Original', resized)

while(True):
    l_h = cv2.getTrackbarPos('LH', 'Values')
    l_s = cv2.getTrackbarPos('LS', 'Values')
    l_v = cv2.getTrackbarPos('LV', 'Values')
    u_h = cv2.getTrackbarPos('UH', 'Values')
    u_s = cv2.getTrackbarPos('US', 'Values')
    u_v = cv2.getTrackbarPos('UV', 'Values')

    mask = cv2.inRange(hsv, (l_h, l_s, l_v), (u_h, u_s, u_v))

    cv2.imshow('Circles', resized)
    cv2.imshow('Masked', mask)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()