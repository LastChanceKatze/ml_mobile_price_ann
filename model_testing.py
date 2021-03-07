import preprocess as pp
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping
import visualize as vs
import numpy as np
import evaluate as ev


def predict(model, x_test):
    # predict test data
    y_pred = model.predict(x=x_test)
    y_pred = y_pred.argmax(axis=1)
    return y_pred


# preprocess data
x_train, x_test, y_train, y_test = pp.preprocess_data()

# shape and num classes
input_shape = x_train.shape[1]
num_classes = len(np.unique(y_train))

##################################################################################################
""" 2 layers - 128 neurons """
# model definition
model = keras.Sequential([
    layers.Dense(units=128, activation='relu', input_shape=[input_shape]),
    layers.Dense(units=128, activation='relu'),
    layers.Dense(units=num_classes, activation='softmax')])

print(model.summary())
# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# early stopping
es = EarlyStopping(monitor='val_loss', mode='auto', patience=10, verbose=1)

# fit
history = model.fit(x=x_train, y=y_train, epochs=200, batch_size=64, verbose=1,
                    callbacks=[es],
                    validation_data=(x_test, y_test))

vs.plot_acc_history(history.history)
vs.plot_loss_history(history.history)

# evaluate model
preds = model.evaluate(x=x_test, y=y_test, return_dict=True)
print(preds)

y_pred = predict(model, x_test)
ev.evaluate_model(y_test, y_pred)
ev.confusion_matrix(y_test, y_pred, class_names=None, plot=False)
##################################################################################################
