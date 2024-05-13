import numpy as np
a = np.load('D:\COCO_train2014_000000000560.jpg.npy')
for i, array in enumerate(a):
    print(f"Array {i+1}:")
    print(array)