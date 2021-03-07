import matplotlib.pyplot as plt


def plot_acc_history(history):
    plt.figure(figsize=(9, 5))
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Accuracy through epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='best')
    plt.show()


def plot_loss_history(history):
    plt.figure(figsize=(9, 5))
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Loss through epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='best')
    plt.show()

