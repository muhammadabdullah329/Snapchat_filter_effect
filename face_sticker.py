import cv2
import numpy as np
import dlib
from math import hypot

#cap = cv2.VideoCapture('vid.mp4')
cap = cv2.VideoCapture(0)
nose_img = cv2.imread('nose.png')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# loop until the end of program
while True:

    success, img = cap.read()
    #img = cv2.resize(img, (0, 0), None, 0.4, 0.4)
    imgoriginal = img.copy()

    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = detector(imgGray)

# loop for finding all faces in the image
    for face in faces:
        x1, y1 = face.left(),face.top()
        x2, y2 = face.right(), face.bottom()
        #imgoriginal = cv2.rectangle(img,(x1, y1),(x2, y2),(0,255,0),2)
        landmarks = predictor(imgGray,face)
        myPoints = []
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            myPoints.append([x,y])
            #cv2.circle(imgoriginal, (x, y), 2, (0, 0, 255), cv2.FILLED)
            #cv2.putText(imgoriginal,str(n), (x,y-2), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5,(0,0,255),1)

        myPoints = np.array(myPoints)

        # extracting features from face

        bbox = cv2.boundingRect(myPoints[29:36])
        x, y, w, h = bbox

        nose_img = cv2.resize(nose_img, (w, h))
        nose_img_gray = cv2.cvtColor(nose_img, cv2.COLOR_BGR2GRAY)
        #nose_img_blur = cv2.GaussianBlur(nose_img_gray,(9,9),0)
        _, nose_mask = cv2.threshold(nose_img_gray,25,255,cv2.THRESH_BINARY_INV)

        #imgoriginal = cv2.rectangle(imgoriginal,(x,y),(x+w,y+h),(255,0,0),3)

        cropimg = imgoriginal[y:y+h, x:x+w]

        cropimg_mask = cv2.bitwise_and(cropimg, cropimg, mask=nose_mask)

        finalimg = cv2.add(cropimg_mask,nose_img)

        imgoriginal [y:y+h, x:x+w] = finalimg

        #cv2.imshow("nose mask", nose_mask)
        #cv2.imshow("nose", cropimg)
        #cv2.imshow("nose1", cropimg_mask)
        #cv2.imshow("final",finalimg)


    cv2.imshow("original",imgoriginal)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break