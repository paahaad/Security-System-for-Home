import cv2


face_cascade = cv2.CascadeClassifier("C:\\Users\\HARSH\\PycharmProjects\\machineLearnig\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")


img1 = cv2.imread("C:\\Users\\HARSH\\PycharmProjects\\machineLearnig\\venv\\Scripts\\test3.JPG",1)

img = cv2.resize(img1,(int(img1.shape[1]/3),int(img1.shape[0]/3)))

gry_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


face_img = face_cascade.detectMultiScale(gry_img,scaleFactor=1.04,minNeighbors=50)

for x,y,w,h in face_img:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

img_resized = cv2.resize(img,(int(img.shape[1]/3),int(img.shape[0]/3)))

cv2.imshow("Name",img_resized)

cv2.waitKey(0)
cv2.destroyAllWindows()
