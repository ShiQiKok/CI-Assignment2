from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from xlsxwriter import Workbook
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def visualise(mlp):
    ''' Function to plot the topology of the neural network.'''

    # get number of neurons in each layer
    n_neurons = [len(layer) for layer in mlp.coefs_]
    n_neurons.append(mlp.n_outputs_)

    # calculate the coordinates of each neuron on the graph
    y_range = [0, max(n_neurons)]
    x_range = [0, len(n_neurons)]
    loc_neurons = [[[l, (n+1)*(y_range[1]/(layer+1))] for n in range(layer)] for l,layer in enumerate(n_neurons)]
    x_neurons = [x for layer in loc_neurons for x,y in layer]
    y_neurons = [y for layer in loc_neurons for x,y in layer]

    # identify the range of weights
    weight_range = [min([layer.min() for layer in mlp.coefs_]), max([layer.max() for layer in mlp.coefs_])]

    # prepare the figure
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title("Model Architecture")
    # draw the neurons
    ax.scatter(x_neurons, y_neurons, s=100, zorder=5)
    # draw the connections with line width corresponds to the weight of the connection
    for l,layer in enumerate(mlp.coefs_):
        for i,neuron in enumerate(layer):
            for j,w in enumerate(neuron):
                ax.plot([loc_neurons[l][i][0], loc_neurons[l+1][j][0]], [loc_neurons[l][i][1], loc_neurons[l+1][j][1]], 'white', linewidth=((w-weight_range[0])/(weight_range[1]-weight_range[0])*5+0.2)*1.2)
                ax.plot([loc_neurons[l][i][0], loc_neurons[l+1][j][0]], [loc_neurons[l][i][1], loc_neurons[l+1][j][1]], 'grey', linewidth=(w-weight_range[0])/(weight_range[1]-weight_range[0])*5+0.2)


def plot_ROC(gold_standard, prediction):
    ''' Function to plot the Receiver Operating Characteristic (ROC) with
        true positive rate against false positive rate.'''

    ## Obtain the false positive rate and true positive rate
    fpr, tpr, _ = roc_curve(gold_standard, prediction)

    ## Plot the ROC Curve
    plt.figure()
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0,1], [0,1], linestyle='--', color='black') ## major diagonal
    plt.plot(fpr, tpr, color='orange', label='area={:.2f}'.format(auc(fpr, tpr)))
    plt.fill_between(fpr,tpr, 0, facecolor='bisque')
    plt.legend()
    plt.show()


def test_one_hidden_layer():
    ''' Testing function to find the accuracy for the neural network of
        1 hidden layer with different numbers of hidden neurones. The output will
        be written to 'result.xlsx' file. '''

    # Workbook Initialization
    wb = Workbook('result.xlsx')
    sheet = wb.add_worksheet('1_layer')
    sheet.write('A1', 'Number of Neurones')
    sheet.write('B1', 'Accuracy')
    row = 1

    for n in range(1, 12):
        mlp = MLPClassifier(hidden_layer_sizes=(n), max_iter=1000,random_state=1)
        mlp.fit(X_train, y_train)
        predictions = mlp.predict(X_test)
        cr = classification_report(y_test, predictions,output_dict=True)
        accuracy = round(cr["accuracy"],2)
        print(f"layer: 1, neurones: {n}")
        print(accuracy)

        # Write to file
        sheet.write(row, 0, n)
        sheet.write(row, 1, accuracy)
        row += 1

    wb.close()


def test_two_hidden_layers():
    ''' Testing function to find the accuracy for the neural network of
        2 hidden layers with different numbers of hidden neurones. The output will
        be written to 'result.xlsx' file. '''

    # Workbook Initialization
    wb = Workbook('result.xlsx')
    sheet = wb.add_worksheet('2_layers')
    sheet.write('A1', 'Number of Neurones')
    sheet.write('B1', 'Accuracy')
    row = 1

    for i in range(1, 12):
        for j in range(1, 12):
            mlp = MLPClassifier(hidden_layer_sizes=(i, j), max_iter=1000,random_state=1)
            mlp.fit(X_train, y_train)
            predictions = mlp.predict(X_test)
            cr = classification_report(y_test, predictions,output_dict=True)
            accuracy = round(cr["accuracy"],2)
            print(f"layer:2, neurones: ({i},{j})")
            print(accuracy)

            # Write to file
            sheet.write(row, 0, f'{i}, {j}')
            sheet.write(row, 1, accuracy)
            row += 1

    wb.close()

def test_activation(sizes=(3,4)):
    ''' Testing function to find the accuracy for the neural network of
        different activation functions. The output will be written to
        'result.xlsx' file. '''

    # Workbook Initialization
    wb = Workbook('result.xlsx')
    sheet = wb.add_worksheet('activation')
    sheet.write('A1', f'Topology: {sizes}')
    sheet.write('A2', 'Activation')
    sheet.write('B2', 'Accuracy')
    activation=["identity", "logistic", "tanh", "relu"]
    row = 2

    for a in activation:
        mlp = MLPClassifier(hidden_layer_sizes=sizes, max_iter=1000, activation=a, random_state=1)
        mlp.fit(X_train, y_train)
        predictions = mlp.predict(X_test)
        cr = classification_report(y_test, predictions,output_dict=True)
        accuracy = round(cr["accuracy"],2)
        print(f"activation: {a}, neurones: {sizes}")
        print(accuracy)

        # Write to file
        sheet.write(row, 0, a)
        sheet.write(row, 1, accuracy)
        row += 1

    wb.close()

## ======================== MAIN =========================
if __name__ == "__main__":
    # Read the dataset
    data = pd.read_csv('heart.csv')
    # List all columns having categorical values
    categorical_input = ['Sex' , 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

    ## Encode the categorical data to numerical data
    oe = OrdinalEncoder()
    oe.fit(data[categorical_input])
    data[categorical_input] = oe.transform(data[categorical_input])
    # print(oe.categories_)

    ## Drop the row where RestingBP = 0
    data.drop(data[data['RestingBP'] == 0].index, inplace=True)

    ## Impute value 0 for the Cholesterol column
    ### replace value 0 with NaN
    data['Cholesterol'].replace(0, np.nan, inplace=True)
    imputer = KNNImputer()
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    X = data.drop(columns='HeartDisease')
    Y = data['HeartDisease']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=1)

    ## Scale the input
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    ## Construct Multilayer Neural Network
    mlp = MLPClassifier(hidden_layer_sizes=(3, 4), activation='relu',max_iter=1000, random_state=1)
    mlp.fit(X_train, y_train)
    predictions = mlp.predict(X_test)

    ## Evaluate Performance
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))

    ## Visualisation
    visualise(mlp)
    plot_ROC(y_test, predictions)

    ## Parameter Testing
    # test_one_hidden_layer()
    # test_two_hidden_layers()
    # test_activation()
