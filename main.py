import numpy as np
import os
import matplotlib.pyplot as plt

IMG_NUMBER = 10000
# IMG_NUMBER = 30

def load_data(filename):
  array = []
  with open(os.path.join(os.path.dirname(__file__), filename), "r") as f:
    for re in range(IMG_NUMBER):
      t = []
      i = 0
      try:
        while True:
          dot = int(f.readline().replace("\n","").replace("+","").replace(".0ns",""))
          if dot > 1000000:
            continue
          t.append(dot)
          i += 1
          if i == 625:
            break
        npa = np.array([t])
        max = npa.max()
        min = npa.min()
        npa = ((npa-min)/(max+0.0000000001-min))
        npa = np.reshape(npa,(25,25,1))
        array.append(npa)

        # plt.imshow(np.reshape(npa,(25,25)),cmap='gray', vmin=0, vmax=1)
        # plt.savefig(str(re)+str(filename)+".jpg")

      except Exception as e:
        break
  return np.array(array)

a_imgs = load_data("htcp.tr")
b_imgs = load_data("hybla.tr")
train_data = np.concatenate([a_imgs[:1000], b_imgs[:1000]], axis=0)
test_data = np.concatenate([a_imgs[1000:1600], b_imgs[1000:1600]], axis=0)

a_train_label = [0]*1000
a_test_label = [0]*600
b_train_label = [1]*1000
b_test_label = [1]*600

a_train_label.extend(b_train_label)
train_labels = a_train_label
a_test_label.extend(b_test_label)
test_labels = a_test_label

import keras
from keras import layers
from keras import models
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(25,25,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))
model.summary()

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=30, batch_size=10)

test_loss, test_acc = model.evaluate(test_data, test_labels)
print(test_acc)



