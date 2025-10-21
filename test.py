import cv2

imgSRC = './lab_images/robotSofia.jpg'

img = cv2.imread(imgSRC)
cv2.imshow('Robot Sofia', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
