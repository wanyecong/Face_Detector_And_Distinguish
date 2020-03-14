import tensorflow as tf
from tensorflow import keras

def build_model(len_class=4,input_shape=(100,100,3)):
    model=tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(0.01), input_shape=input_shape),
        tf.keras.layers.Conv2D(32,(3,3),activation='relu',),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
        tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32,activation='relu'),
        tf.keras.layers.Dense(len_class,activation='softmax')

    ])
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
    return model
def build_train_data(father_dir):
    train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255,
                                                                 rotation_range=40,
                                                                 width_shift_range=0.1,
                                                                 height_shift_range=0.1,
                                                                 shear_range=0.1,
                                                                 zoom_range=0.1,
                                                                 horizontal_flip=False,
                                                                 fill_mode='nearest')
    train_generator = train_datagen.flow_from_directory(father_dir,
                                                        target_size=(100, 100),
                                                        batch_size=32,)
    return train_generator
def build_test_data(father_dir):
    test_datagen=keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
    test_generator=test_datagen.flow_from_directory(father_dir,target_size=(100,100),batch_size=32)
    return test_generator