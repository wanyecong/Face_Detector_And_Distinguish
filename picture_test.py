'''
I wanted to get data by crawing Baidu pictures,But he does so well in Anti reptile that I can't crawing
his pictures in my current capacity.Ten i try another one "sougoupicture" ,it is much easier.
I just want use this project to exercise my ability well.
1.craw picture
2.get faces by opencv
3.exercise a model to class the faces
4.debug and adjust
5.successful
Let's begin~~~~~~~~~~~
'''
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import freetype
#define the face picture info
Picture_Height=64
Picture_Width=64
model= tf.keras.models.load_model('model.h5')
image1=cv2.imread('test3.jpg')
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
rects = detector.detectMultiScale(image1, scaleFactor=1.1,
                                      minNeighbors=2, minSize=(20, 20),
                                      flags=cv2.CASCADE_SCALE_IMAGE)
for (x, y, w, h) in rects:
    namelist=np.array(['林允儿','李现','易烊千玺','章若楠'])
    image=cv2.resize(image1[y:y+h,x:x+w],(100,100))
    cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 255, 0),2)
    image=image.reshape(1,100,100,3)
    #translate into float and /255
    image = image.astype('float32')
    image /= 255
    #get predict
    name=namelist[model.predict_classes(image)]
    name=str(name)
    image_pil=Image.fromarray(cv2.cvtColor(image1,cv2.COLOR_BGR2RGB))
    draw=ImageDraw.Draw(image_pil)
    font = ImageFont.truetype("simkai.ttf", w//5, encoding="utf-8")
    draw.text((x,y-w//5),name,font=font,fill=(0,255,0))
    image1=cv2.cvtColor(np.asarray(image_pil),cv2.COLOR_RGB2BGR)
'''this can only show english'''
# image1=cv2.putText(image1, str(name),#the type of the name most be translated into str although the original type is str
#                     (x + 30, y - 30),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     1,
#                     (255, 0, 255),
#                     2)
cv2.imshow('Test picture',image1)
cv2.waitKey(0)
