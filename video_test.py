import numpy as np
import tensorflow as tf
import cv2
from PIL import Image,ImageDraw,ImageFont
import freetype
model= tf.keras.models.load_model('model.h5')
cap=cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
namelist=np.array(['林允儿','李现','易烊千玺','章若楠'])
while (True):
    ret,frame=cap.read()
    rects = detector.detectMultiScale(frame, scaleFactor=1.1,
                                      minNeighbors=2, minSize=(20, 20),
                                      flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in rects:
        #circle the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0),2)
        #capture the face and translate into array
        image=cv2.resize(frame[y:y+h,x:x+w],(100,100))
        #image=tf.image.resize(fig1,(100,100))
        image=image.reshape(1,100,100,3)
        #translate into float and /255
        image = image.astype('float32')
        image /= 255
        #get predict
        name=namelist[model.predict_classes(image)]
        name = str(name)
        image_pil = Image.fromarray(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)
        font = ImageFont.truetype("simkai.ttf", 50, encoding="utf-8")
        draw.text((x, y - 50), name, font=font, fill=(0, 255, 0))
        image1 = cv2.cvtColor(np.asarray(image_pil), cv2.COLOR_RGB2BGR)
        # cv2.putText(frame, str(name),# chinese is not supported  we can use freetype
        #             (x + 30, y + 30),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             1,
        #             (255, 0, 255),
        #             2)
        cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()