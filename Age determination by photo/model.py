import pandas as pd
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50


def load_train(path):
    # Целевые классы
    labels = pd.read_csv(os.path.join(path, 'labels.csv'))
    directory = os.path.join(path, 'final_files/')
    # Изображения (приводим нормировку 0 - 1, добавляем доп.данные путем аугментации, выделяем 25% на валидацию)
    datagen = ImageDataGenerator(rescale=1. / 255,
                                 horizontal_flip=True,
                                 validation_split=0.25)
    train_datagen_flow = datagen.flow_from_dataframe(
        dataframe=labels,
        directory=directory,
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        subset='training',
        seed=12345)
    return train_datagen_flow


def load_test(path):
    # Целевые классы
    labels = pd.read_csv(os.path.join(path, 'labels.csv'))
    directory = os.path.join(path, 'final_files/')
    # Изображения (приводим нормировку 0 - 1, добавляем доп.данные путем аугментации, выделяем 25% на валидацию)
    datagen = ImageDataGenerator(rescale=1. / 255,
                                 validation_split=0.25)
    val_datagen_flow = datagen.flow_from_dataframe(
        dataframe=labels,
        directory=directory,
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        subset='validation',
        seed=12345)
    return val_datagen_flow


def create_model(input_shape):
    backbone = ResNet50(input_shape=input_shape,
                        weights='/datasets/keras_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                        include_top=False)

    optimizer = Adam(lr=0.001)

    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='relu'))
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

    return model


def train_model(model, train_datagen_flow, valid_datagen_flow, batch_size=None, epochs=15,
                steps_per_epoch=None, validation_steps=None, verbose=2):
    if steps_per_epoch is None:
        steps_per_epoch = len(train_datagen_flow)
    if validation_steps is None:
        validation_steps = len(valid_datagen_flow)

    model.fit(train_datagen_flow,
              validation_data=valid_datagen_flow,
              batch_size=batch_size,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=verbose,
              epochs=epochs)

    return model
