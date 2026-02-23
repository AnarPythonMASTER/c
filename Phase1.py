from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import Accuracy, F1Score, Precision, Recall
from keras.optimizers import SGD, Adam, Adagrad, RMSprop, Nadam, AdamW
from google.colab import drive
drive.mount('/content/drive/')
import os
directory_path = r'/content/drive/MyDrive/31 dekabr/Deep Learning'
if os.path.isdir(directory_path):
    folder_contents = os.listdir(directory_path)
    print(f"Contents of '{directory_path}':")
    for item in folder_contents:
        print(item)
else:
    print(f"Error: '{directory_path}' is not a valid directory.")
import tensorflow as tf
from tensorflow import keras

DATA_DIR = r"/content/drive/MyDrive/31 dekabr/Deep Learning"
IMG_SIZE = (224, 168)
BATCH = 32
SEED = 42

# 1) Train split
train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,# 80%% train
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH,
    color_mode="rgb"# ensures (H,W,3)
)

#2) Validation split
val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,# same split rule
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH,
    color_mode="rgb"
)

print("Classes:", train_ds.class_names)
num_classes = len(train_ds.class_names)

# 3) Normalize
train_ds = train_ds.map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y))
val_ds   = val_ds.map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y))

#4) Speed, this is not thatimportant but recommended
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.prefetch(tf.data.AUTOTUNE)
for images, labels in train_ds.take(1):
    print(images.shape)

from keras.src.losses import loss
from keras.src.optimizers import optimizer
model= keras.Sequential([layers.Input(shape=(224,168,3)),#224,168,3
                  layers.Conv2D(filters=32,kernel_size=(3,3), padding="same",activation="relu",kernel_initializer="he_normal",strides=(1,1)),#parameter size is 3x3x3x32 +32bias=896 parameters in 1st layer
                  layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),#112x84x32 <- this is the output size,
                  layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding="same",activation="relu",kernel_initializer="he_normal"),#3x3x32x64+64=18496
                  layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),#56x42x64 <-output,
                  layers.Conv2D(filters=128,kernel_size=(3,3), strides=(1,1),activation="relu",kernel_initializer="he_normal",padding="same"),#3x3x64x128+128=73856
                  layers.MaxPool2D(pool_size=(2,2),strides=(2,2)), #28x21x128<-outpute
                  # layers.Conv2D(filters=256,kernel_size=(3,3),padding="same",activation="relu",kernel_initializer="he_normal", strides=(1,1)),
                  # layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
                  # layers.Conv2D(filters=512,kernel_size=(3,3), padding="same",activation="relu",kernel_initializer="he_normal",strides=(1,1)),
                  # layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
                  # layers.Conv2D(filters=1024,kernel_size=(3,3), padding="same",activation="relu",kernel_initializer="he_normal",strides=(1,1)),
                  # layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
                  # layers.Conv2D(filters=2048,kernel_size=(3,3), padding="same",activation="relu",kernel_initializer="he_normal",strides=(1,1)),
                  # layers.MaxPool2D(pool_size=(2,2),strides=(2,3)),
                  # layers.Conv2D(filters=4096,kernel_size=(3,3), padding="same",activation="relu",kernel_initializer="he_normal",strides=(1,1)),
                  # layers.MaxPool2D(pool_size=(2,2),strides=(2,1)),#19x19 after this maxpool
                  layers.Flatten(),#75264 input dimensions
                  # layers.Dense(512,activation="relu",kernel_initializer="he_normal"),#512x361+512bias
                  # layers.Dense(1024,activation="relu",kernel_initializer="he_normal"),#1024x512+1024
                  layers.Dense(64,activation="relu",kernel_initializer="he_normal"),#64x75264 + 64 bias=4816960 parameters in the 1st layer of te MLP
                  layers.Dense(21,activation="softmax")])#21x64 +21 bias = 1365 PARAMETERS IN THE LAST LAYER
model.compile(optimizer=SGD(learning_rate=0.0001,momentum=0.9, nesterov=True),
              loss=SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

early=EarlyStopping(monitor="val_loss",
                    patience=15,
                    restore_best_weights=True)

history = model.fit(train_ds,
          validation_data=val_ds,
          epochs=30,
          callbacks=early)

loss, acc = model.evaluate(val_ds)
print("Validation Loss:", loss)
print("Validation Accuracy:", acc)