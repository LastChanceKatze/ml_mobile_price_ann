import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(y_test, y_pred):
    print("Report:\n", metrics.classification_report(y_test, y_pred))
    print('F1 Score:\n', metrics.f1_score(y_test, y_pred, average='micro'))
    print('Precision Score:\n', metrics.precision_score(y_test, y_pred, average="micro"))
    print("Accuracy:\n", metrics.accuracy_score(y_test, y_pred))


def confusion_matrix(y_test, y_pred, class_names, plot=False):
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", conf_matrix)

    if plot:
        df_conf = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
        sns.heatmap(df_conf, annot=True, fmt='g')
        plt.xlabel("False")
        plt.ylabel("True")
        plt.show()
