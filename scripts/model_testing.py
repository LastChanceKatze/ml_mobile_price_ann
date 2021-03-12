import preprocess as pp
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import visualize as vs
import numpy as np
import evaluate as ev


def predict(model, x_test):
    # predict test data
    y_pred = model.predict(x=x_test)
    y_pred = y_pred.argmax(axis=1)
    return y_pred


def fit(x_train, y_train, x_test, y_test, callback):
    # fit
    history = model.fit(x=x_train, y=y_train, epochs=100, batch_size=32, verbose=1,
                        callbacks=[callback, ModelCheckpoint(filepath="best_model.h5",
                                                       monitor="val_loss",
                                                       save_best_only=True)],
                        validation_data=(x_test, y_test))

    vs.plot_history(history.history)


def evaluate(model, x_test, y_test):
    # evaluate model
    preds = model.evaluate(x=x_test, y=y_test, return_dict=True)
    print(preds)

    y_pred = predict(model, x_test)
    ev.evaluate_model(y_test, y_pred)
    ev.confusion_matrix(y_test, y_pred, class_names=None, plot=False)


# preprocess data
x_train, x_test, y_train, y_test = pp.preprocess_data()

# shape and num classes
input_shape = x_train.shape[1]
num_classes = len(np.unique(y_train))

# early stopping
es = EarlyStopping(monitor='val_loss', mode='auto', patience=10, verbose=1)
##################################################################################################
""" 2 layers - 128 neurons """
""" bactch = 32, epochs = 50, opt = rmsprop """


def create_network(optimizer):
    # model definition
    model = keras.Sequential([
        layers.Dense(units=128, activation='relu', input_shape=[input_shape]),
        layers.Dense(units=128, activation='relu'),
        layers.Dense(units=num_classes, activation='softmax')])

    # print(model.summary())
    # compile the model
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

"""
GRID SEARCH TEST
classifier = KerasClassifier(build_fn=create_network, verbose=0)

params = {
    'optimizer': ['adam', 'rmsprop', 'sgd'],
    'batch_size': [32, 64, 100],
    # 'callbacks': [es],
    # 'validation_data': [(x_test, y_test)]
}

grid = GridSearchCV(estimator=classifier, param_grid=params)
grid_result = grid.fit(x_train, y_train)
print(grid_result.best_params_)

model = create_network('rmsprop')
"""
#
# model = create_network('adam')
#
# # fit
# history = model.fit(x=x_train, y=y_train, epochs=100, batch_size=32, verbose=1,
#                     callbacks=[es],
#                     validation_data=(x_test, y_test))
#
# vs.plot_history(history.history)
#
# # evaluate model
# preds = model.evaluate(x=x_test, y=y_test, return_dict=True)
# print(preds)
#
# y_pred = predict(model, x_test)
# ev.evaluate_model(y_test, y_pred)
# ev.confusion_matrix(y_test, y_pred, class_names=None, plot=False)
##################################################################################################
""" 2x64 - adam """
""" acc - 0.92 """
""" sgd bolji """
# # model
# model = keras.Sequential([
#     layers.Dense(units=64, activation='relu', input_shape=[input_shape]),
#     layers.Dense(units=64, activation='relu'),
#     layers.Dense(units=num_classes, activation='softmax')])
#
# # print(model.summary())
# # compile the model
# model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# fit(x_train, y_train, x_test, y_test, es)
# model = keras.models.load_model("best_model.h5")
# evaluate(model, x_test, y_test)
##################################################################################################
""" 2x32 - adam """
""" acc - 0.95 """
# # model
# model = keras.Sequential([
#     layers.Dense(units=32, activation='relu', input_shape=[input_shape]),
#     layers.Dense(units=32, activation='relu'),
#     layers.Dense(units=num_classes, activation='softmax')])
#
# # print(model.summary())
# # compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# fit(x_train, y_train, x_test, y_test, es)
# model = keras.models.load_model("best_model.h5")
# evaluate(model, x_test, y_test)
##################################################################################################
""" 128, 64 - adam """
""" acc - 0.91 """
# # model
# model = keras.Sequential([
#     layers.Dense(units=128, activation='relu', input_shape=[input_shape]),
#     layers.Dense(units=64, activation='relu'),
#     layers.Dense(units=num_classes, activation='softmax')])
#
# # compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# fit(x_train, y_train, x_test, y_test, es)
# model = keras.models.load_model("best_model.h5")
# evaluate(model, x_test, y_test)
##################################################################################################
""" 32, 64 - adam """
""" slicni rez za druge optimizatore """
""" acc - 0.93 """
# # model
# model = keras.Sequential([
#     layers.Dense(units=64, activation='relu', input_shape=[input_shape]),
#     layers.Dense(units=32, activation='relu'),
#     layers.Dense(units=num_classes, activation='softmax')])
#
# # compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# fit(x_train, y_train, x_test, y_test, es)
# model = keras.models.load_model("best_model.h5")
# evaluate(model, x_test, y_test)
##################################################################################################
""" 128, 64, 32 - adam - nije lose"""
""" sgd - manji overfit - slican acc """
""" acc - 0.93 """
# # model
# model = keras.Sequential([
#     layers.Dense(units=32, activation='relu', input_shape=[input_shape]),
#     layers.Dense(units=64, activation='relu'),
#     layers.Dense(units=128, activation='relu'),
#     layers.Dense(units=num_classes, activation='softmax')])
#
# # compile the model
# model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# fit(x_train, y_train, x_test, y_test, es)
# model = keras.models.load_model("best_model.h5")
# evaluate(model, x_test, y_test)
##################################################################################################
""" 64, 32, 16 - adam """
""" acc - 0.92 """
# # model
# model = keras.Sequential([
#     layers.Dense(units=16, activation='relu', input_shape=[input_shape]),
#     layers.Dense(units=32, activation='relu'),
#     layers.Dense(units=64, activation='relu'),
#     layers.Dense(units=num_classes, activation='softmax')])
#
# # compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# fit(x_train, y_train, x_test, y_test, es)
# model = keras.models.load_model("best_model.h5")
# evaluate(model, x_test, y_test)
##################################################################################################
""" 32, 32, 32 - adam """
""" acc - 0.89 """
# # model
# model = keras.Sequential([
#     layers.Dense(units=32, activation='relu', input_shape=[input_shape]),
#     layers.Dense(units=32, activation='relu'),
#     layers.Dense(units=32, activation='relu'),
#     layers.Dense(units=num_classes, activation='softmax')])
#
# # compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# fit(x_train, y_train, x_test, y_test, es)
# model = keras.models.load_model("best_model.h5")
# evaluate(model, x_test, y_test)
##################################################################################################
""" 64, 64, 64 - adam """
""" acc - 0.90 """
# # model
# model = keras.Sequential([
#     layers.Dense(units=64, activation='relu', input_shape=[input_shape]),
#     layers.Dense(units=64, activation='relu'),
#     layers.Dense(units=64, activation='relu'),
#     layers.Dense(units=num_classes, activation='softmax')])
#
# # compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# fit(x_train, y_train, x_test, y_test, es)
# model = keras.models.load_model("best_model.h5")
# evaluate(model, x_test, y_test)
##################################################################################################
""" 128, 128, 128 - adam - overfit"""
""" acc - 0.90 """
# # model
# model = keras.Sequential([
#     layers.Dense(units=128, activation='relu', input_shape=[input_shape]),
#     layers.Dense(units=128, activation='relu'),
#     layers.Dense(units=128, activation='relu'),
#     layers.Dense(units=num_classes, activation='softmax')])
#
# # compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# fit(x_train, y_train, x_test, y_test, es)
# model = keras.models.load_model("best_model.h5")
# evaluate(model, x_test, y_test)
##################################################################################################
""" 2x32 - adam + Dropout + regulizers - stabilniji algoritam"""
""" acc - 0.93 """
# # model
# model = keras.Sequential([
#     layers.Dense(units=32, activation='relu', input_shape=[input_shape],
#                  kernel_regularizer=regularizers.l2(0.001)),
#     layers.Dropout(0.3),
#     layers.Dense(units=32, activation='relu',
#                  kernel_regularizer=regularizers.l2(0.001)),
#     layers.Dropout(0.3),
#     layers.Dense(units=num_classes, activation='softmax')])
#
# # print(model.summary())
# # compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# fit(x_train, y_train, x_test, y_test, es)
# model = keras.models.load_model("best_model.h5")
# evaluate(model, x_test, y_test)
##################################################################################################
""" 32, 64 - adam + Dropout + regularizatori -> stabilnija mreza"""
""" slicni rez za druge optimizatore """
""" acc - 0.93 """
# model
model = keras.Sequential([
    layers.Dense(units=32, activation='relu', input_shape=[input_shape],
                 kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dropout(0.3),
    layers.Dense(units=64, activation='relu',
                 kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dropout(0.3),
    layers.Dense(units=num_classes, activation='softmax')])

# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

fit(x_train, y_train, x_test, y_test, es)
model = keras.models.load_model("best_model.h5")
evaluate(model, x_test, y_test)
