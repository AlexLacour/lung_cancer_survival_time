from keras.applications.resnet import ResNet50
from keras.layers import Input, Dropout, Dense, BatchNormalization, Activation, Concatenate, Flatten
from keras.models import Model
import keras.backend as K
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import metrics_t9gbvr2 as metric


clinicals_dir = './x_train/features/clinical_data.csv'
y_dir = './y_train/output.csv'
radiomics_dir = './x_train/features/radiomics.csv'
archives_dir = './x_train/images'

archives = [np.load(os.path.join(
    archives_dir, archive)) for archive in os.listdir(archives_dir)]

clinicals = pd.read_csv(clinicals_dir, index_col='PatientID').sort_index()
radiomics = pd.read_csv(radiomics_dir, index_col='PatientID').sort_index()
time_and_events = pd.read_csv(y_dir, index_col='PatientID').sort_index()

clinicals = pd.get_dummies(clinicals, drop_first=True)

y = time_and_events['SurvivalTime']
events = time_and_events['Event']

for col in radiomics.columns:
    n_na = radiomics[col].isna().sum()
    if(n_na == radiomics.shape[0]):
        radiomics.drop(col, axis=1, inplace=True)

radiomics = pd.get_dummies(radiomics, drop_first=True)

full_scans = []
full_masks = []
full_clinicals = []
full_radiomics = []
full_y = []
full_events = []
idx = []

"""
CT SCANS => pixels are Hounsfield Units (usually from -1000 to +3000)
"""
min_val = -1024
max_val = 3072

for archive, clinical, radiomic, time, event, i in zip(archives, clinicals.values, radiomics.values, y.values, events.values, y.index):
    for scan, mask in zip(archive['scan'], archive['mask']):
        rescaled_scan = (scan - min_val) / (max_val - min_val)
        rescaled_scan = np.asarray(rescaled_scan, dtype='float32')
        rescaled_scan = cv2.cvtColor(rescaled_scan, cv2.COLOR_GRAY2RGB)
        full_scans.append(rescaled_scan)
        full_masks.append(mask)
        full_clinicals.append(clinical)
        full_radiomics.append(radiomic)
        full_y.append(time)
        full_events.append(event)
        idx.append(i)

full_scans = np.asarray(full_scans)
full_radiomics = np.asarray(full_radiomics)
full_clinicals = np.asarray(full_clinicals)
full_events = np.asarray(full_events)
full_y = np.asarray(full_y)
full_y = np.reshape(full_y, (len(full_y), 1))

scan_train, scan_test, mask_train, mask_test, c_train, c_test, r_train, r_test, y_train, y_test, events_train, events_test, idx_train, idx_test = train_test_split(
    full_scans, full_masks, full_clinicals, full_radiomics, full_y, full_events, idx, test_size=0.2, random_state=0)


def negative_log_likelihood(events):
    def loss(y_true, y_pred):
        hazard_ratio = K.exp(y_pred)
        log_risk = K.log(hazard_ratio)
        uncensored_likelihood = y_pred - log_risk
        censored_likelihood = uncensored_likelihood * events
        neg_likelihood = -K.sum(censored_likelihood)
        return neg_likelihood
    return loss


def plot_loss(h):
    plt.figure()
    plt.plot(h.history['loss'])
    plt.show()


def create_model(input_shape=(92, 92, 3), n_radiomics=len(r_train[0]), n_clinicals=len(c_train[0])):
    """
    SCANS
    """
    scans_input = ResNet50(input_shape=input_shape, include_top=False)
    scans_features = Flatten()(scans_input.output)
    scans_features = Dropout(0.5)(scans_features)
    scans_features = Dense(64)(scans_features)
    scans_features = BatchNormalization()(scans_features)
    scans_features = Activation('relu')(scans_features)

    """
    RADIOMICS DATA
    """
    rad_input = Input(shape=(n_radiomics,))
    rad_features = Dense(n_radiomics)(rad_input)
    rad_features = BatchNormalization()(rad_features)
    rad_features = Dropout(0.5)(rad_features)
    rad_features = Activation('relu')(rad_features)

    """
    CLINICALS DATA
    """
    cl_input = Input(shape=(n_clinicals,))
    cl_features = Dense(n_clinicals)(cl_input)
    cl_features = BatchNormalization()(cl_features)
    cl_features = Dropout(0.5)(cl_features)
    cl_features = Activation('relu')(cl_features)

    hazard = Concatenate()([scans_features, rad_features, cl_features])
    hazard = Dropout(0.2)(hazard)
    hazard = Dense(64)(hazard)
    hazard = BatchNormalization()(hazard)
    hazard = Activation('relu')(hazard)
    hazard = Dense(10)(hazard)
    model_output = Dense(1)(hazard)
    model_output = Activation('linear')(model_output)

    model = Model(inputs=[scans_input.input, rad_input,
                          cl_input], outputs=model_output)
    model.compile(optimizer='rmsprop',
                  loss=negative_log_likelihood(events_train))

    model.summary()

    return model


if __name__ == '__main__':
    model = create_model()

    h = model.fit([scan_train, r_train, c_train], y_train,
                  epochs=20,
                  shuffle=False,
                  batch_size=32)
    model.save_weights('weights.h5')
    plot_loss(h)
    model.load_weights('weights.h5')
    print("MODEL LOADED")

    hr_pred = model.predict([scan_train, r_train, c_train])
    hr_pred = np.exp(hr_pred)

    hr_pred2 = model.predict([scan_test, r_test, c_test])
    hr_pred2 = np.exp(hr_pred2)

    train_true = pd.DataFrame()
    train_true['PatientID'] = idx_train
    train_true['SurvivalTime'] = y_train
    train_true['Event'] = events_train
    train_true.set_index('PatientID', inplace=True)

    train_pred = pd.DataFrame()
    train_pred['PatientID'] = idx_train
    train_pred['SurvivalTime'] = hr_pred
    train_pred['Event'] = None
    train_pred.set_index('PatientID', inplace=True)

    test_true = pd.DataFrame()
    test_true['PatientID'] = idx_test
    test_true['SurvivalTime'] = y_test
    test_true['Event'] = events_test
    test_true.set_index('PatientID', inplace=True)

    test_pred = pd.DataFrame()
    test_pred['PatientID'] = idx_test
    test_pred['SurvivalTime'] = hr_pred2
    test_pred['Event'] = None
    test_pred.set_index('PatientID', inplace=True)

    ci_train = metric.cindex(train_true, train_pred)
    ci_test = metric.cindex(test_true, test_pred)

    print(ci_train)
    print(ci_test)
