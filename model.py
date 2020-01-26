from keras.models import Model
from keras.layers import Dense, Conv2D, UpSampling2D, Input
from keras.applications.densenet import DenseNet169


def create_model(input_shape=(92, 92, 1)):
    img_input = Input(shape=input_shape)
    features = DenseNet169(include_top=False)(img_input)
    return Model(inputs=img_input, outputs=features)


if __name__ == '__main__':
    create_model().summary()
