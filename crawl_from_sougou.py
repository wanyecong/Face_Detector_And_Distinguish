import requests
import re
from urllib.parse import quote
from pathlib import Path
name_lists=['张天爱','迪丽热巴','古力娜扎','李沁','章若楠','林允儿','邓伦','肖战','王一博','李现','易烊千玺','权志龙']
name_lists_english=['zhangtianai','dilireba','gulinazha','liqin','zhangruonan','linyuner','denglun','xiaozhan','wangyibo','lixian',
                    'yiyangqianxi','quanzhilong']
def load_picture(i):
    path='G:/FaceDetectProject/original_datasets/'+str(name_lists_english[i])
    if not Path(path).exists():
        path1='original_datasets/'+str(name_lists_english[i])
        new_path=Path('G:/FaceDetectProject/')/path1
        new_path.mkdir(parents=True)#make dir


    try:
        encodingname=quote(name_lists[i],encoding='utf-8')
        url='https://pic.sogou.com/pics?query='+str(encodingname)+'&di=2&_asf=pic.sogou.com&w=05009900'
        response=requests.get(url)
        html=response.text
        re_format=re.compile('"thumbUrl":"(.*?)",')
        picture_urls=re_format.findall(html)
    except:
        print("The picture of {} can't be download".format(str(name_lists[i])) )


    if len(picture_urls)!=0:
        for url in range(len(picture_urls)):
             with open("G:/FaceDetectProject/original_datasets/"+str(name_lists_english[i])+'/'+str(name_lists_english[i])+str(url)+".jpg", 'wb') as f:
                 row=requests.get(picture_urls[url]).content
                 f.write(row)



