import cv2
import numpy as np
import dlib
from scipy.spatial import distance as dist
import time

def eyeAspectRatio(points):
	A = dist.euclidean(points[1], points[5])
	B = dist.euclidean(points[2], points[4])
	C = dist.euclidean(points[0], points[3])
	
	return (A + B) / (2.0 * C)

def getROI(frame, image, landmarks, eye):
	if eye == 0:
		points = [36, 37, 38, 39, 40, 41]
	else:
		points = [42, 43, 44, 45, 46, 47]
		
	region = np.array([[landmarks.part(point).x, landmarks.part(point).y] for point in points])
	margin = 7
	
	left = np.min(region[:, 0])
	top = np.min(region[:, 1])
	right = np.max(region[:, 0])
	bottom = np.max(region[:, 1])
	
	height = abs(top - bottom)
	width = abs(left - right)	
	grayEye = image[top:bottom, left+margin:right-margin]
	roi = frame[top:bottom, left+margin:right-margin]
	thresh = calibrate(grayEye)
	_, threshEye = cv2.threshold(grayEye, thresh, 255, cv2.THRESH_BINARY)
	prepEye = preprocess(threshEye)
	x, y = getIris(prepEye, roi)
	#text = str((x*left)/(width*100.0))
	#cv2.putText(frame, text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
	#print(height)
	cv2.circle(frame, (x+left, y+top), 3, (0, 255, 0), -1)
	
	ear = eyeAspectRatio(region)
	
	return (x*left)/(width*100.0), (y*top)/(height*100.0), ear

def getSize(eye, t):
	height, width = eye.shape
	_, thresh = cv2.threshold(eye, t, 255, cv2.THRESH_BINARY)
	n_pixels = height*width
	#print(n_pixels)
	
	black_pixels = n_pixels - cv2.countNonZero(thresh)
	#print("->", black_pixels)
	try:
		ratio = black_pixels * 1.0 / n_pixels
		return ratio
	except ZeroDivisionError:
		return None
	

def calibrate(eye):
	iris_size = 0.48
	trials = {}
	
	for t in range(5, 100, 5):
		trials[t] = getSize(eye, t)
	
	try:
		best_threshold, size = min(trials.items(), key = lambda x : abs(x[1] - iris_size))
		#print(best_threshold, size)
		return best_threshold
	except TypeError:
		return None
	

def preprocess(image):
	kernel = np.array([[0., 1., 0.], [1., 2., 1.], [0., 1., 0.]], dtype = np.uint8)
	blur = cv2.bilateralFilter(image, 5, 10, 10)
	#leftEroded = cv2.erode(leftBlur, kernel, iterations = 1) 
	dilated = cv2.dilate(blur, kernel)
	return cv2.bitwise_not(dilated)


def getIris(image, roi):
	contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	all_contours = sorted(contours, key = cv2.contourArea, reverse = True)
	margin = 5
	#return max_contour
	#for contour in contours: 
	#	cv2.drawContours(roi, contour, -1, (255, 0, 0), 2)
	#cv2.drawContours(roi, max_contour, -1, (255, 0, 0), 2)
	try:
		max_contour = all_contours[0]
		M = cv2.moments(max_contour)
		x = int(M['m10'] / M['m00']) + margin
		y = int(M['m01'] / M['m00'])
		roi = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
		cv2.circle(roi, (x, y), 3, (0, 0, 255), -1)
		#cv2.imshow("ROI", roi)
		return x, y
	except (IndexError, ZeroDivisionError):
		return 0, 0
		
	
	
def printText(frame, text):
	width, height, _ = frame.shape
	cv2.putText(frame, text, (width // 2, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

if __name__ == "__main__":
	cap = cv2.VideoCapture(0)
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
	total = 0
	previousRatio = 1
	while True:
		retr, frame = cap.read()
		frame = cv2.flip(frame, 1)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		try:
			faces = detector(gray)
		
			landmarks = predictor(gray, faces[0])
			#cv2.circle(frame, (landmarks.part(0).x, landmarks.part(1).y), 3, (255, 0, 0), -1)
		except: 
			continue
		margin = 7
		Lhori, Lverti, Lear = getROI(frame, gray, landmarks, 0)
		
		Rhori, Rverti, Rear = getROI(frame, gray, landmarks, 1)
		
		avgEAR = (Lear + Rear) / 2.0
		
		avgHori = (Lhori + Rhori) / 2.0
		avgVerti = (Lverti + Rverti) / 2.0
		
		if avgHori < 0.8:
			printText(frame, "LEFT")
		elif avgHori > 1.70:
			printText(frame, "RIGHT")
		elif avgVerti < 0.60:
			printText(frame, "UP")
		else:
			printText(frame, "CENTER")
		
		if(avgEAR < 0.20):
			if(previousRatio >= 0.20):
				total += 1
		previousRatio = avgEAR
			
		cv2.putText(frame, "Counter: " + str(total), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
		
		cv2.imshow("Frame", frame)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			cap.release()
			cv2.destroyAllWindows()	
