import pathlib
import cv2 as cv

cascade_path = pathlib.Path(cv.__file__).parent.absolute()/"data/haarcascade_frontalface_default.xml"
print(cascade_path)

classifier = cv.CascadeClassifier(str(cascade_path))

capture  = cv.VideoCapture(1)
# capture = cv.VideoCapture("C:\\Users\\Arqam Nisar\\Downloads\\1.mp4")


while True:
    _, frame = capture.read()
    grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    faces = classifier.detectMultiScale(
        grayscale,
        scaleFactor = 1.1,
        minNeighbors = 7,
        minSize = (30, 30),
        flags = cv.CASCADE_SCALE_IMAGE
        
    )
    
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        
    cv.imshow("Face Detected", frame)
    if cv.waitKey(1) == ord('q'):
        break
    

capture.release()
cv.destroyAllWindows()
        
