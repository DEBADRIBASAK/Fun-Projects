import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
import PIL.Image as im
import os

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--root",type=str,required=False,default="/home/debadri/Pictures")
	parser.add_argument("--path",type=str,required=True,default=None)
	args = parser.parse_args()
	path = os.path.join(args.root,args.path)
	try:
		img = cv2.imread(path,0)
	except Error:
		print("Could not read",Error)
		quit()
	print("img shape = ",img.shape)
	#print("imf shape = ",imf.shape)
	# RGB to YCbCr

	# img_ycc = img.convert("YCbCr")
	# img_ycc = np.ndarray((img.size[1],img.size[0],3),'u1',img_ycc.tobytes())

	# DCT 

	img1 = img.astype(float)#-128.0
	imf = np.zeros(img.shape)
	for i in range(img.shape[0]//8):
		for j in range(0,img.shape[1]//8,1):
			imf[i*8:(i+1)*8,j*8:(j+1)*8] = cv2.dct(img1[i*8:(i+1)*8,j*8:(j+1)*8])

	# Quantization

	m = np.max(imf)*0.01
	mask = abs(imf)>=m
	imf1 = imf*mask

	# Inverse DCT

	img2 = np.zeros(img.shape)
	for i in range(0,img.shape[0]//8,1):
		for j in range(0,img.shape[1]//8,1):
			img2[i*8:(i+1)*8,j*8:(j+1)*8] = cv2.idct(imf1[i*8:(i+1)*8,j*8:(j+1)*8])

	#img1+=128.0



	# cv2.imshow("Original",img)
	# cv2.imshow("DCT",imf)
	# cv2.imshow("Quantized",imf1)
	# cv2.imshow("Compressed",img2.astype(int))
	# #cv2.waitKey()
	# print("img1 = ",img1)
	# print("imf1 = ",imf1)
	# print("img2 = ",img2.astype(int).astype(float))
	cv2.imwrite(os.path.join(args.root,"img2.png"),img2.astype(int))
	cv2.imwrite(os.path.join(args.root,"imf.png"),imf1.astype(int))
	plt.imshow(np.hstack((img,imf,imf1,img2)),cmap='gray')
	plt.show()

main()