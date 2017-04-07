import cv2
print(cv2.__version__)

cascade_src = 'cars.xml'
video_src = '/home/peter/Downloads/BWZ 7450.mp4'
#video_src = 'dataset/video2.avi'

def nothing(*arg):
    pass

cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)

cv2.namedWindow('edge')
cv2.createTrackbar('thrs1', 'edge', 50, 255, nothing)
cv2.createTrackbar('thrs2', 'edge', 200, 255, nothing)

while True:
    ret, img = cap.read()
    if (type(img) == type(None)):
        break
    
    thrs1 = cv2.getTrackbarPos('thrs1', 'edge') * 2
    thrs2 = cv2.getTrackbarPos('thrs2', 'edge')

    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,thrs1,thrs2,0)
    image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    imgout = cv2.drawContours(img, contours, -1, (0,255,0), 1)

    cv2.imshow('edge',imgout)

    
    if cv2.waitKey(33) == 27:
        break
    if ch == 13:
        step = True
    if ch == 32:
        paused = not paused

cv2.destroyAllWindows()
