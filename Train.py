import tensorflow as tf
import keras
from keras import backend as K
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tqdm.auto import tqdm



from DataGenerator import DataGenerator
from BuildModel import build_model

BATCH_SIZE=10
IMG_SIZE=512
model_path='model/cnn.h5'

train_gen = DataGenerator('./TrainData', BATCH_SIZE, IMG_SIZE, aug=True)
test_gen = DataGenerator('./TestData', BATCH_SIZE, IMG_SIZE, aug=False)




############## Loss function ##############
def loss_function(y_true, y_pred):
  squared_diff = tf.square(y_true - y_pred)
  return tf.reduce_mean(squared_diff)
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coef(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss
##########################################



K.clear_session()

opt = keras.optimizers.Adam(learning_rate=1e-3)
#loss_function
try:
    model = load_model(model_path)
except:
    model = build_model(IMG_SIZE)


model.compile(optimizer=opt,loss=bce_dice_loss,metrics=[dice_coef])



############# call back ##############
mc = ModelCheckpoint(filepath= model_path, 
                    monitor='val_dice_coef', 
                    mode='max', 
                    save_best_only=True,
                    save_weights_only=False)
rl = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.5, patience=8)
es = EarlyStopping(monitor='val_loss', mode='min', patience=15)
# tb = TensorBoard(log_dir=model_dir)
callbacks_list = [mc, rl, es]

################ Fit ######################
model.fit(train_gen,
          validation_data = test_gen,
          batch_size=BATCH_SIZE, epochs=1000,
          callbacks=callbacks_list)
