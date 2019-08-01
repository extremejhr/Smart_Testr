# USAGE
# python text_detection.py --image images/lebron_james.jpg --east frozen_east_text_detection.pb

# import the necessary packages

import numpy as np
import time
import cv2


def text_detector(image, width, height, min_confidence):

	# set the new width and height and then determine the ratio in change
	# for both the width and height
	(newW, newH) = (width, height)

	# resize the image and grab the new image dimensions
	image = cv2.resize(image, (newW, newH))
	(H, W) = image.shape[:2]

	text_flag = 0

	# define the two output layer names for the EAST detector model that
	# we are interested -- the first is the output probabilities and the
	# second can be used to derive the bounding box coordinates of text
	layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

	# load the pre-trained EAST text detector
	net = cv2.dnn.readNet("frozen_east_text_detection.pb")

	# construct a blob from the image and then perform a forward pass of
	# the model to obtain the two output layer sets
	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)

	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)

	# show timing information on text prediction

	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]

	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the geometrical
		# data used to derive potential bounding box coordinates that
		# surround text

		scoresData = scores[0, 0, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability, ignore it

			if scoresData[x] >= min_confidence:

				text_flag = 1

				break

	return text_flag
