import numpy as np
import matplotlib.pyplot as plt
import cv2
import SimpleITK as sitk

imdir = './x_test/images/'

if __name__ == '__main__':
    archive = np.load(imdir + 'patient_000.npz')
    scans = archive['scan']
    masks = archive['mask']
    print(np.max(scans))
    print(np.min(scans))
