import cv2
import numpy as np

cap = cv2.VideoCapture(0)

frames = []

for i in range(10):
	_,frame = cap.read()
	frames.append(frame)
background = np.array(frames).mean(axis=0)
print("size = ",background.shape)

while(1):
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([0,0,10])
    upper_red = np.array([15,255,255])
    
    mask = cv2.inRange(frame, lower_red, upper_red)
    mask1 = 1-mask
    bg = cv2.bitwise_and(background,background,mask=mask)
    fg = cv2.bitwise_and(frame,frame, mask= mask1)
    res = cv2.bitwise_or(bg,fg,mask=np.ones(bg.shape))

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
