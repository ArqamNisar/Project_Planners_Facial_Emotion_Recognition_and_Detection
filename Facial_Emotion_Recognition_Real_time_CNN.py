#Load the important libraries:
import cv2 as cv
import numpy as np
import pathlib
import warnings
warnings.filterwarnings("ignore")
from keras.models import model_from_json
from keras.utils import img_to_array

emotions_total = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy',4:'Neutral',5:'Sad', 6:'Surprise'}

#Load Json and create recog model:
recog_json_file = open('models/recog_model.json', 'r')
loaded_model_json = recog_json_file.read()
recog_json_file.close()
recog_model = model_from_json(loaded_model_json)

#Load the pre-trained model:
recog_model.load_weights("models/recog_model.h5")
print("Loading model from saved files...")

# capture = cv.VideoCapture(1)
capture = cv.VideoCapture("C:\\Users\\Arqam Nisar\\Downloads\\1.mp4")

while True:
    retrn, test_image = capture.read()
    test_image = cv.resize(test_image, (1280, 720))
    
    if not retrn:
        break
    
    cascade_path = pathlib.Path(cv.__file__).parent.absolute()/"data/haarcascade_frontalface_default.xml"

    face_detected = cv.CascadeClassifier(str(cascade_path))  
    # face_detected = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    grayscale_img = cv.cvtColor(test_image, cv.COLOR_BGR2GRAY)  
    
    faces = face_detected.detectMultiScale(grayscale_img, scaleFactor= 1.3, minNeighbors = 5)
    for (x, y, w, h) in faces:
        cv.rectangle(test_image, (x, y-50), (x + w, y + h + 10), (255,0,0), thickness=3)
        
        #Taking out ROI from image:
        ROI_img = grayscale_img[y:y+h, x:x+w] 
        ROI_img = cv.resize(ROI_img, (48,48))
        img_pixels = img_to_array(ROI_img)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255
        
        preds = recog_model.predict(img_pixels)
        
        #Finding max probability of the class image belongs to:
        max_ind = np.argmax(preds[0])
        
        cv.putText(test_image, emotions_total[max_ind], (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 1,(0,255,0), 2)

    resize_img = cv.resize(test_image,(1000, 700))
    cv.imshow('Facial Emotion Recognition using CNN' , resize_img)
    
    #Wait until Q is pressed:
    if cv.waitKey(10) == ord('q'):
        break
    
capture.release()
cv.destroyAllWindows()