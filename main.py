import numpy as np

if __name__ == '__main__':
    archive = np.load('patient_000.npz')
    scan = archive['scan']
    mask = archive['mask']
