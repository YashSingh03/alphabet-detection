#importing all the important modules.
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os
import ssl
import time


X = np.load('c122_project.npz')['arr_0']
y = pd.read_csv("c122_project.csv")["labels"]
print(pd.Series(y).value_counts())

classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
           "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
nclasses = len(classes)

#splitting the data and scaling it
X_train, X_test, y_train, y_test = train_test_split(X, y ,random_state = 9, train_size = 7500, test_size = 2500)

#scalling the features
X_train_scale = X_train / 255.0
X_test_scale = X_test / 255.0

#fitting the traning data into the model
clf = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(X_train_scale, y_train)

#calculating the accuracy of the model
y_pred = clf.predict(X_test_scale)
print("Accuracy: ", accuracy_score(y_test, y_pred))


#starting the camera

#Starting the webcam
cap = cv2.VideoCapture(0)

while (True):
    try:
        ret, frame = cap.read()
        #our operations on the frame come here
        gray = cv2.CvtColor(frame, cv2.COLOR_BGR2GRAY)
        #drawing the box in the center of the video
        height, width = gray.shape
        upper_left = (int(width / 2 - 56), int(height / 2 - 56))
        bottom_right = (int(width / 2 + 56), int(height / 2 + 56))
        cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 2)

        #To only consider the area inside the box for detecting the digit
        #roi = Region Of Interest
        roi = gray[upper_left[1]:bottom_right[1],
                   upper_left[0]:bottom_right[0]]

        #Converting cv2 image to pil format
        im_pil = Image.fromarray(roi)

        #convert to gray scale image(L means each pixel is represented by a single value form 0 to 255)
        image_bw = im_pil.convert('L')
        img_bw_resized = image_bw.resize((28, 28), Image.ANTIALIAS)

        #imverting the image
        img_bw_resized_inverted = PIL.ImageOps.invert(img_bw_resized)
        pixel_filter = 20

        #converting to scalaer quantity
        min_pixel = np.percentile(img_bw_resized_inverted, pixel_filter)

        #using clip to limit the values between 0 to 255
        img_bw_resized_inverted_scaled = np.clip(
            img_bw_resized_inverted - min_pixel, 0, 255)
        max_pixel = np.max(img_bw_resized_inverted)

        #converting into the array
        img_bw_resized_inverted_scaled = np.asarray(
            img_bw_resized_inverted_scaled) / max_pixel

        #creating a test sample and making a prediction
        test_sample = np.array(img_bw_resized_inverted_scaled).reshape(1, 784)
        test_pred = clf.predict(test_sample)
        print("the predicted class is ", test_pred)

        #displaying the resulted frame
        cv2.imshow('frame', gray)
        if cv2.waitkey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        pass

#when everything done release the camera
cap.release()
cv2.destroyAllWindows()

