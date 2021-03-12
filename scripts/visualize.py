import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model


def plot_history(history):
    fig, axs = plt.subplots(1, 2)

    axs[0].plot(history['accuracy'])
    axs[0].plot(history['val_accuracy'])
    axs[0].set(xlabel='Epoch', ylabel='Accuracy')
    axs[0].title.set_text('Accuracy through epochs')
    axs[0].legend(['Train', 'Test'], loc='best')

    axs[1].plot(history['loss'])
    axs[1].plot(history['val_loss'])
    axs[1].set(xlabel='Epoch', ylabel='Loss')
    axs[1].title.set_text('Loss through epochs')
    axs[1].legend(['Train', 'Test'], loc='best')
    plt.show()


# TODO: visualize network model
def plot_nn(model, filename):
    plot_model(model, show_shapes=True, to_file=filename)
