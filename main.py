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
from crawl_from_sougou import load_picture
from get_face_picture import Detect_face_and_load
name_lists=['张天爱','迪丽热巴','古力娜扎','李沁','章若楠','林允儿','邓伦','肖战','王一博','李现','易烊千玺','权志龙']
name_lists_english=['zhangtianai','dilireba','gulinazha','liqin','zhangruonan','linyuner','denglun','xiaozhan','wangyibo','lixian',
                    'yiyangqianxi','quanzhilong']


for i in range(len(name_lists)):
     load_picture(i)# crawl the stars' picture and saved
#there are some problem when I try  ,eventually,I succeed after 4 times debug

for i in range(len(name_lists)):
   Detect_face_and_load(i)#get everyones' faces and saved
#there are some problem when I try,eventually,I succeed after one day debug
'''
===================================================================================================================
=Originally I want to distinguish the twelve stars but the  size of the dataset is so small that the model can not  @
=be trained well .Finally,i chose four good faces to train.And it can get 87% of score.                             @
=This is a good ending                                                                                              @
=====================================================================================================================
'''



'''
runing the train_python.py
After we finished the training , we can use the picture_test   and video_test to test our model  
I found it is very good
'''