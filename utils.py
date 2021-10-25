import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle


data_dir = "./train/train"

categories = ['barbijo', 'normal']

data = []


def getData():
    for category in categories:
        path = os.path.join(data_dir, category)
        label = categories.index(category)

        for nomImage in os.listdir(path):
            pathImage = os.path.join(path, nomImage)
            image = cv2.imread(pathImage)

            try:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (224, 224))
                image = np.array(image, dtype=np.float32)
                data.append([image, label])
            except Exception as e:
                pass
    pik = open('data.pickle', 'wb')
    pickle.dump(data, pik)
    pik.close()


getData()


def loadData():
    pick = open('data.pickle', 'rb')
    data = pickle.load(pick)
    pick.close()

    np.random.shuffle(data)

    feature = []
    labels = []

    for img, label in data:
        feature.append(img)
        labels.append(label)
    feature = np.array(feature, dtype=np.float32)
    labels = np.array(labels)

    feature = feature/255.0

    return [feature, labels]
