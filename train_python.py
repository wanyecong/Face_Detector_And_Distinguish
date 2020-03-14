import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from crawl_from_sougou import load_picture
from Draw_model import *
from get_face_picture import name_lists

Batchsize=32
Picture_shape=(100,100)
Channel=3
if __name__=='__main__':
    model=build_model()
    train_path="G:/FaceDetectProject/data"
    train_data=build_train_data(train_path)
    test_path='G:/FaceDetectProject/testdata'
    test_data=build_test_data(test_path)
    history=model.fit_generator(train_data,
                                steps_per_epoch=len(train_data)//32,
                                epochs=100,validation_data=test_data)#There will be a long time

    model.save('model.h5')#save the model

    fig=plt.figure(figsize=(16,6))
    plt.plot(history.epoch,history.history.get('loss'),label='loss')
    plt.plot(history.epoch, history.history.get('acc'), label='acc')
    plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')
    plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')
    plt.legend()
    plt.show()
