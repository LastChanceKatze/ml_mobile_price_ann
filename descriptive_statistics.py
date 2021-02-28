import matplotlib.pyplot as plt
import seaborn as sns
import preprocess as pp

def plot_density_graph(data):
    for column in data.columns:
        fig = plt.figure()
        data[column].plot(kind="density", figsize=(17, 17))

        plt.vlines(data[column].mean(), ymin=0, ymax=0.5, linewidth=0.5)

        plt.vlines(data[column].median(), ymin=0, ymax=0.5, linewidth=2.0, color="red")

        plt.savefig("./density_graphs/density_" + str(column)+".jpg")

def plot_boxplot(data):
    data.drop('price_range', axis=1).plot(kind='box', figsize=(10, 10), subplots=True, layout=(3,7),
                                      sharex=False, sharey=False,
                                      title='Box Plot for each input variable')
    plt.subplots_adjust(wspace=0.5)
    plt.savefig('./graphs/features_box_plot.jpg')
   # plt.show()

def plot_barplots(data):
    num_rows = 4
    num_cols = 5
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20,20))
    plt.subplots_adjust(hspace=0.5, wspace=0.3, top=0.95)

    for i in range(0, num_rows):
        for j in range(0, num_cols):
            index = j * num_rows + i
            col_name = data.columns[index]
            sns.barplot(x='price_range', y=col_name, data=data, ax=axs[i][j])

    plt.savefig("./graphs/bar_plot.jpg")
    #plt.show()

def plot_histogram(data, data_column):
    plt.figure(figsize=(7, 4))
    data[data_column].plot(kind='hist', figsize=(4, 4))
    # sns.histplot(data_column, kde=True)
    plt.savefig("./graphs/" + data_column + "_histogram.jpg")
   # plt.show()

def plot_count(data, data_column):
    plt.figure(figsize=(7, 4))
    sns.countplot(x=data_column, data=data)
    plt.savefig("./graphs/" + data_column + "_countplot.jpg")
   # plt.show()

def correlation_matrix(data):
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, ax=ax)
    plt.savefig("./graphs/correlation_matrix.png")
    #plt.show()

    corr = abs(corr['price_range']).sort_values(ascending=False)
    return corr

def describe_data(data):


    df1=data.iloc[:, 0 : 10]
    df2=data.iloc[:, 10 : 20]

    print("Describe data (columns 0-9): \n" + str(df1.describe(include='all')))
    print("----------------------------------------------------------")

    print("Describe data: (columns 10-19): \n" + str(df2.describe(include='all')))
    print("----------------------------------------------------------")

    print("Print median: \n" + str(data.median()))
    print("----------------------------------------------------------")

def generate_graphs():

    data = pp.preprocess_data_for_desc_stat()
    plot_density_graph(data)
    plot_boxplot(data)
    plot_barplots(data)
    plot_histogram(data=data, data_column='price_range')
    plot_count(data=data, data_column='price_range')
    correlation_matrix(data)

generate_graphs()