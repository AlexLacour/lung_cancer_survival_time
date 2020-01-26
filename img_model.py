from keras.models import Sequential
from keras.layers import UpSampling2D, Conv2D, LeakyReLU, MaxPooling2D
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator


def addUpProject(model, n_filters):
    model.add(UpSampling2D())
    model.add(Conv2D(int(n_filters), kernel_size=3, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(int(n_filters), kernel_size=3, padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    return model


def create_model(input_shape=(92, 92, 1)):
    model = Sequential()
    model.add(Conv2D(256, kernel_size=3, padding='same', input_shape=input_shape))

    for _ in range(2):
        model.add(MaxPooling2D())
        model.add(Conv2D(256, kernel_size=3, padding='same'))

    for _ in range(2):
        model.add(UpSampling2D())
        model.add(Conv2D(256, kernel_size=3, padding='same'))

    model.add(Conv2D(1, kernel_size=3, padding='same'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy')

    return model


if __name__ == '__main__':
    model = create_model()
    model.summary()
    imdir = './x_train/images/'
    archives_names = os.listdir(imdir)
    print(archives_names)
    X = []
    y = []
    for archive_name in archives_names:
        archive = np.load(imdir + archive_name)
        for scan, mask in zip(archive['scan'], archive['mask']):
            X.append(scan)
            y.append(mask)
    X = np.reshape(X, (len(X), 92, 92, 1))
    y = np.reshape(y, (len(y), 92, 92, 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    train_gen = ImageDataGenerator()
    test_gen = ImageDataGenerator()

    train_gen = train_gen.flow(X_train, y_train)
    test_gen = test_gen.flow(X_test, y_test)

    h = model.fit_generator(train_gen,
                            epochs=50,
                            validation_data=test_gen)

    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
