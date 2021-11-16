import matplotlib.pyplot as plt
import tensorflow as tf
import visualkeras
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import random
import math

np.random.seed(46)

from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.initializers import he_normal, zeros, glorot_normal, RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.regularizers import l1, l2, l1_l2

from matplotlib import style
import pathlib
import shutil

from tensorflow.keras.callbacks import Callback
from keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import f1_score, precision_score, recall_score, auc
import datetime
import time

np.random.seed(46)
tf.random.set_seed(46)

# Dividing data into training and validation set
# We have taken 1/4 of the randomly chosen training dataset for the purpose of validation
# ****************************************************************************************
# code to make Validation folder to be executed only once

val_dir = "Validation"
# os.mkdir(val_dir)

n_classes = 43
train_dir = "Train{0}"

# Moving files from train to validation directory  to be executed only once
# for n in tqdm(range(n_classes)):
#     path = os.path.join(val_dir, str(n))
#     os.mkdir(path)
#     src_path = train_dir.format('/' + str(n))
#     files = os.listdir(src_path)
#     rand_idx = random.sample(range(len(files)), math.ceil(len(files)/4))
#     for idx in rand_idx :
#         src = src_path + "/" + files[idx]
#         shutil.move(src, path)


# Setting up the size of the image and channel variables

IMG_WIDTH = 30
IMG_HEIGHT = 30
N_CHANNELS = 3
BATCH_SIZE = 32
N_EPOCHS = 30
VAL_BATCH_SIZE = 32
CLASS_NAMES = list(range(43))
N_CLASSES = 43  # check with previous variable too

train_path = "Train"  # Path for Train Data Set
val_path = "Validation"  # Path for Validation Dataset

# Path of validation and train data set
data_root_train = pathlib.Path(train_path)
data_root_val = pathlib.Path(val_path)

# Getting path of all the individual images of training and validation dataset
all_image_paths_train = list(data_root_train.glob('*/*'))
all_image_paths_train = [str(path) for path in all_image_paths_train]
all_image_paths_val = list(data_root_val.glob('*/*'))
all_image_paths_val = [str(path) for path in all_image_paths_val]

# print(all_image_paths_val)

# Count of number of images in training and validation set
image_count_train = len(all_image_paths_train)
image_count_val = len(all_image_paths_val)

print('number of images in training data set after splitting - {}'.format(image_count_train))
print('number of images in validation dataset after splitting - {}'.format(image_count_val))
#################################


# Now extracting labels for the images present in validation and train dataset

label_names_train = sorted(int(item.name) for item in data_root_train.glob('*/') if item.is_dir())
label_names_val = sorted(int(item.name) for item in data_root_val.glob('*/') if item.is_dir())
label_to_index_train = dict((name, index) for index, name in enumerate(label_names_train))
label_to_index_val = dict((name, index) for index, name in enumerate(label_names_val))
all_image_labels_train = [label_to_index_train[int(pathlib.Path(path).parent.name)] for path in all_image_paths_train]
all_image_labels_val = [label_to_index_val[int(pathlib.Path(path).parent.name)] for path in all_image_paths_val]

# print(label_names_train)
# print(label_names_val)
# print(label_to_index_train)
# print(label_to_index_train)
# print(all_image_labels_train)
# print(all_image_labels_val)


# Updating the bounding coordinates of the images

# *******************************************************************************************************
# Loading Training information CSV dataframe
df_train = pd.read_csv("Train.csv")
# Updating coordinates

for idx, row in df_train.iterrows():
    W = row['Width']
    H = row['Height']
    if W > IMG_WIDTH:
        diff = W - IMG_WIDTH
        df_train.iloc[idx, 4] = df_train.iloc[idx]['Roi.X2'] - diff
    else:
        diff = IMG_WIDTH - W
        df_train.iloc[idx, 4] = df_train.iloc[idx]['Roi.X2'] + diff
    if H > IMG_HEIGHT:
        diff = H - IMG_HEIGHT
        df_train.iloc[idx, 5] = df_train.iloc[idx]['Roi.Y2'] - diff
    else:
        diff = IMG_HEIGHT - H
        df_train.iloc[idx, 5] = df_train.iloc[idx]['Roi.Y2'] + diff

# creating dataframes for training and validation dataset

train_idx_list = []
val_idx_list = []
# n = len(df_train)
# for i in range(n):
#     if df_train.iloc[i]["Path"] in all_image_paths_train:
#         train_idx_list.append(i)
#
#
#
# print("you are here")
# print(len(train_idx_list))
# print(df_train['Path'])
# print("you are here")
# print(len(all_image_paths_train), len(df_train))
# df_train['path_exists'] = df_train.Path.isin(all_image_paths_train).astype(int)
# print(df_train['path_exists'].value_counts())


for path_tr in tqdm(all_image_paths_train):
    try:
        train_idx_list.append(df_train[df_train['Path'] == path_tr[:]].index[0])
    except IndexError:

        print("error due to", path_tr[:])

for path_val in tqdm(all_image_paths_val):
    path_val2 = "Train/" + path_val[11:]

    try:
        val_idx_list.append(df_train[df_train['Path'] == path_val2].index[0])
    except IndexError:
        print("error due to", path_val)

# print(val_idx_list)
new_df_train = pd.DataFrame()
new_df_val = pd.DataFrame()

new_df_train = new_df_train.append(df_train.iloc[train_idx_list], ignore_index=True)
new_df_val = new_df_val.append(df_train.iloc[val_idx_list], ignore_index=True)

new_df_train = new_df_train.drop(['Height', 'Width', 'ClassId', 'Path'], axis=1)
new_df_val = new_df_val.drop(['Height', 'Width', 'ClassId', 'Path'], axis=1)


# ******************************************************************************************************************

# Defining function to generate data by increasing brightness in darker images and increasing contrast

def tfdata_generator(images, labels, df, training, batch_size=32):
    def func_preprocess(filename, labels, df):
        # for preprocessing the images

        # reading the path
        image_string = tf.io.read_file(filename)

        image = tf.image.decode_png(image_string, channels=N_CHANNELS)

        # converting the image to float value between 0 to 1 so as analyse its current brightness and contrast level
        image = tf.image.convert_image_dtype(image, tf.float32)

        # adjusting contrast and brightness of the image
        if tf.math.reduce_mean(image) < 0.3:
            image = tf.image.adjust_contrast(image, 5)
            image = tf.image.adjust_brightness(image, 0.2)

        # resizing the image
        image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH], method="nearest", preserve_aspect_ratio=False)
        image = image / 255.0

        return image, {"classification": labels, "regression": df}

    # creating a dataset from tensor slices
    dataset = tf.data.Dataset.from_tensor_slices((images, labels, df))

    if training:
        dataset = dataset.shuffle(31000)

    # Transform and batch data at the same time
    dataset = dataset.map(func_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    # prefetch the data into CPU
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


# Training and validation data generator

tf_image_generator_train = tfdata_generator(all_image_paths_train, all_image_labels_train, new_df_train, training=True,
                                            batch_size=32)
tf_image_generator_val = tfdata_generator(all_image_paths_val, all_image_labels_val, new_df_val, training=False,
                                          batch_size=32)


class Border(tf.keras.layers.Layer):

    # to sharpen the borders of the input images
    def __init__(self, num_op):
        super(Border, self).__init__()
        self.num_op = num_op

    def build(self, input_shape):
        self.kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        self.kernel = tf.expand_dims(self.kernel, 0)
        self.kernel = tf.expand_dims(self.kernel, 0)
        self.kernel = tf.cast(self.kernel, tf.float32)

    def call(self, input_):
        return tf.nn.conv2d(input_, self.kernel, strides=[1, 1, 1, 1], padding='SAME')


def create_model():
    input_layer = Input(shape=(IMG_HEIGHT, IMG_WIDTH, N_CHANNELS,), name="input_layer", dtype='float32')

    # Border Layer to make the image clearer
    sharpen = Border(num_op=(IMG_HEIGHT, IMG_WIDTH, N_CHANNELS,))(input_layer)

    # Convolution, maxpool and dropout layers

    convolution_1 = Conv2D(filters=64, kernel_size=(5, 5), activation=relu, kernel_initializer=he_normal(seed=54),
                           bias_initializer=zeros(), name="first_convolutional_layer")(sharpen)

    convolution_2 = Conv2D(filters=128, kernel_size=(3, 3), activation=relu, kernel_initializer=he_normal(seed=55),
                           bias_initializer=zeros(), name="second_convolutional_layer")(convolution_1)

    maxpool_layer1 = MaxPool2D(pool_size=(2, 2), name="first_maxpool_layer")(convolution_2)

    drop_1 = Dropout(0.25)(maxpool_layer1)

    convolution_3 = Conv2D(filters=256, kernel_size=(3, 3), activation=relu, kernel_initializer=he_normal(seed=56),
                           bias_initializer=zeros(), name="third_convolutional_layer")(drop_1)

    maxpool_layer2 = MaxPool2D(pool_size=(2, 2), name="second_maxpool_layer")(convolution_3)

    drop_2 = Dropout(0.25)(maxpool_layer2)

    flat = Flatten(name="flatten_layer")(drop_2)

    # Dense Layer

    dens_1 = Dense(units=512, activation=relu, kernel_initializer=he_normal(seed=45), bias_initializer=zeros(),
                   name="first_dense_layer_classification", kernel_regularizer=l2(0.001))(flat)

    drop_3 = Dropout(0.5)(dens_1)

    classification = Dense(units=43, activation=None, name="classification", kernel_regularizer=l2(0.0001))(drop_3)

    regression = Dense(units=4, activation='linear', name="regression",
                       kernel_initializer=RandomNormal(seed=43), kernel_regularizer=l2(0.1))(drop_3)

    model = Model(inputs=input_layer, outputs=[classification, regression])
    model.summary()
    return model



# Creating custom callback for F1-Score
class Metrics(Callback):

    def __init__(self, validation_data_generator):
        self.validation_data_generator = validation_data_generator

    def on_train_begin(self, logs={}):
        '''
        This function initializes lists to store AUC and Micro F1 scores
        '''
        self.val_f1s = []
        self.val_precisions = []
        self.val_recalls = []
        self.batches = self.validation_data_generator.as_numpy_iterator()

    def on_epoch_end(self, epoch, logs={}):
        '''
        This function calculates the micro f1 and auc scores
        at the end of each epochs
        '''
        current_batch = self.batches.next()
        images = current_batch[0]
        labels = current_batch[1]
        labels = labels["classification"]
        labels = np.array(labels)
        pred = self.model.predict(images)
        pred = pred[0]
        val_predict = (np.asarray(pred)).round()
        idx = np.argmax(val_predict, axis=-1)
        a = np.zeros(val_predict.shape)
        a[np.arange(a.shape[0]), idx] = 1
        val_predict = [np.where(r == 1)[0][0] for r in a]
        val_predict = np.array(val_predict)
        val_targ = labels
        _val_f1 = f1_score(val_targ, val_predict, average='weighted')
        _val_precision = precision_score(val_targ, val_predict, average='weighted')
        _val_recall = recall_score(val_targ, val_predict, average='weighted')
        print("\nEpoch : {0} -  Precision_Score : {1:.2f} - Recall_Score : {2:.2f} - F1_Score : {3:.2f}\n".format(epoch,
                                                                                                                  _val_precision,
                                                                                                                  _val_recall,
                                                                                                                  _val_f1))
        self.val_f1s.append(_val_f1)
        self.val_precisions.append(_val_precision)
        self.val_recalls.append(_val_recall)
        return


# Defining loss functions for classification and regression
# Loss function for bounding box regression
def r2_keras(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


loss = SparseCategoricalCrossentropy(from_logits=True)

# Compiling the model
model = create_model()
lr=0.001
opt = Adam(lr=lr, decay=lr / (epochs * 0.5))
model.compile(optimizer= opt, loss={"classification": loss, "regression": "mse"},
              metrics={"classification": "acc", "regression": r2_keras},
              loss_weights={"classification": 5, "regression": 1})

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True,
                                   write_images=True)
# ModelCheckpoint
NAME = "TrafficSignRecog-first-cut-{0}".format(int(time.time()))
save_best_model = ModelCheckpoint(filepath='best_models/{0}'.format(NAME), monitor='val_loss',
                                  save_best_only=True, mode='min', save_freq='epoch')

# Early stopping to avoide model overfitting
early_stop = EarlyStopping(monitor='loss', min_delta=0.0001, patience=5)
metrics = Metrics(tf_image_generator_val)

# Training
history = model.fit_generator(
    generator=tf_image_generator_train, steps_per_epoch=918,  # train batch size
    epochs=N_EPOCHS,
    validation_data=tf_image_generator_val, validation_steps=306,  # val batch size
    callbacks=[save_best_model, tensorboard_callback, metrics, early_stop]
)

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()



def evaluate_test_images(path, model) :
  """
  Function to make predictions for the test set images
  """
  labels = []
  bbox = []
  all_imgs = os.listdir(path)
  all_imgs.sort()
  for img in tqdm(all_imgs):
    if '.png' in img:
      image_string = tf.io.read_file(path + '/' + img)
      #Loading and decoding image
      image = tf.image.decode_png(image_string, channels=N_CHANNELS)
      #Converting image data type to float
      image = tf.image.convert_image_dtype(image, tf.float32)
      #Adjusting image brightness and contrast
      if tf.math.reduce_mean(image) < 0.3:
        image = tf.image.adjust_contrast(image, 5)
        image = tf.image.adjust_brightness(image, 0.2)
      #Resizing image
      image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH], method="nearest", preserve_aspect_ratio=False)
      image = image/255.0
      image = np.expand_dims(image, axis=0)

      #Predicting output
      pred = model.predict(image)
      labels.append(np.argmax(pred[0][0]))
      bbox.append(pred[1][0])

  return labels, bbox


path = 'Test'
# model = create_model()

labels, bobx = evaluate_test_images(path, model)
print(labels)


test_cv = pd.read_csv('Test.csv')
test_y = test_cv["ClassId"].values

print(test_y)
visualkeras.layered_view(model)
#print("Classification report for RCNN model", metrics.classification_report(test_y, outputs[0]))

print('Test F1 score', f1_score(test_y, labels, average = 'weighted'))
print('Test precision score :', precision_score(test_y, labels, average='weighted'))
print('Test recall score:', recall_score(test_y, labels, average='weighted'))

print('Test Data accuracy: ', accuracy_score(test_y, labels)*100)