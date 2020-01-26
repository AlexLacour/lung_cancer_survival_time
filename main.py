import numpy as np
import matplotlib.pyplot as plt
import cv2

imdir = './x_test/images/'

if __name__ == '__main__':
    archive = np.load(imdir + 'patient_000.npz')
    scan = archive['scan']
    mask = archive['mask']

    for s in scan:
        cv2.imshow('', s)
        cv2.waitKey(0)
    cv2.destroyAllWindow()
