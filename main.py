import preprocess as pp
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import visualize as vs
from sklearn.metrics import confusion_matrix

x_train, x_test, y_train, y_test = pp.preprocess_data()

# model definition
model = keras.Sequential([
    layers.Dense(units=30, activation='relu',
                 kernel_regularizer=regularizers.l2(0.001), input_shape=[x_train.shape[1]]),
    layers.Dropout(0.3),
    layers.Dense(units=50, activation='relu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.3),
    layers.Dense(units=4, activation='softmax')])

print(model.summary())
# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# fit
history = model.fit(x=x_train, y=y_train, epochs=200, batch_size=64, verbose=1,
                    validation_data=(x_test, y_test))

vs.plot_acc_history(history.history)

# evaluate model
preds = model.evaluate(x=x_test, y=y_test, return_dict=True)

# predict test data
y_pred = model.predict(x=x_test)
y_pred = y_pred.argmax(axis=1)

# print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", conf_matrix)

