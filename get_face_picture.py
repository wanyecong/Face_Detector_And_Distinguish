import cv2
from crawl_from_sougou import load_picture
import os
from pathlib import Path
name_lists=['张天爱','迪丽热巴','古力娜扎','李沁','章若楠','林允儿','邓伦','肖战','王一博','李现','易烊千玺','权志龙']

'''opcv can not read chinese but we still need to use chinese to search from sougou picture'''

name_lists_english=['zhangtianai','dilireba','gulinazha','liqin','zhangruonan','linyuner','denglun','xiaozhan','wangyibo','lixian',
                    'yiyangqianxi','quanzhilong']
def Detect_face_and_load(i):
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#define face detector
    new_path = 'G:/FaceDetectProject/Dataset/' + str(name_lists_english[i])
    if not Path(new_path).exists():
        path1 = 'Dataset/' + str(name_lists_english[i])
        most_new_path = Path('G:/FaceDetectProject/') / path1
        most_new_path.mkdir(parents=True)  # make dir


    path='G:/FaceDetectProject/original_datasets/'+str(name_lists_english[i])
    for filename in os.listdir(path):
        image=cv2.imread(path+'/'+filename)
        rects=detector.detectMultiScale(image,scaleFactor=1.1,
                                    minNeighbors=2,minSize=(20,20),
                                    flags=cv2.CASCADE_SCALE_IMAGE)
        if len(rects)!=0:
            (x,y,w,h) = rects[0]
            fig1=cv2.resize(image[y:y+h,x:x+w],(100,100))
            cv2.imwrite("G:/FaceDetectProject/Dataset/"+str(name_lists_english[i])+'/face1'+filename,fig1)
            fig2=cv2.flip(fig1,1)
            cv2.imwrite("G:/FaceDetectProject/Dataset/"+str(name_lists_english[i])+'/face2'+filename,fig2)#flip the picture and get more face data
