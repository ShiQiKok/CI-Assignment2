from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def visualise(mlp):
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
    # draw the neurons
    ax.scatter(x_neurons, y_neurons, s=100, zorder=5)
    # draw the connections with line width corresponds to the weight of the connection
    for l,layer in enumerate(mlp.coefs_):
        for i,neuron in enumerate(layer):
            for j,w in enumerate(neuron):
                ax.plot([loc_neurons[l][i][0], loc_neurons[l+1][j][0]], [loc_neurons[l][i][1], loc_neurons[l+1][j][1]], 'white', linewidth=((w-weight_range[0])/(weight_range[1]-weight_range[0])*5+0.2)*1.2)
                ax.plot([loc_neurons[l][i][0], loc_neurons[l+1][j][0]], [loc_neurons[l][i][1], loc_neurons[l+1][j][1]], 'grey', linewidth=(w-weight_range[0])/(weight_range[1]-weight_range[0])*5+0.2)


data = pd.read_csv('heart.csv')
categorical_input = ['Sex' , 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
# print(data.describe().transpose())

## Encode the categorical data
oe = OrdinalEncoder()
oe.fit(data[categorical_input])
data[categorical_input] = oe.transform(data[categorical_input])
print(oe.categories_)

## Drop the data having RestingBP = 0
data.drop(data[data['RestingBP'] == 0].index, inplace=True)

## Impute value 0 for cholesterol
## replace value 0 to NaN for cholesterol
data['Cholesterol'].replace(0, np.nan, inplace=True)

## KNN Imputer will only impute NaN values
imputer = KNNImputer()
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)


X = data.drop(columns='HeartDisease')
Y = data['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=1)

# Scale the input
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000)
mlp.fit(X_train, y_train)
predictions = mlp.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
# visualise(mlp)

#Test Parameter (HiddenLayerSize & Activation)
neuron = 11
tuples = []
for i in range(1,neuron+1):
    tuples.append((i,))
    for j in range(1,neuron+1):
        tuples.append((i,j))
activation=["identity", "logistic", "tanh", "relu"]

#Create ExcelSheet
import xlsxwriter
workbook = xlsxwriter.Workbook('result(10).xlsx')
worksheet = workbook.add_worksheet("Results")

worksheet.write('A1', 'Hidden Layer Sizes')
worksheet.write('B1', 'Activation')
worksheet.write('C1', 'Average Accuracy')

#Test and write to file
row = 1
average = 10
for t in tuples:
    for a in activation:
        averageAccuracy = 0
        for n in range(average):
            mlp = MLPClassifier(hidden_layer_sizes=t,activation=a, max_iter=1000)
            mlp.fit(X_train, y_train)
            predictions = mlp.predict(X_test)
            cr=classification_report(y_test, predictions,output_dict=True)
            accuracy=round(cr["accuracy"],2)
            averageAccuracy += accuracy
        averageAccuracy /= average
        print("Printing result for activation:{activation} and layer:{layer}\nAverage accuracy:{averageAccuracy}".format(activation=a,layer=t,averageAccuracy=averageAccuracy))
        worksheet.write(row, 0, str(t))
        worksheet.write(row, 1, a)
        worksheet.write(row, 2, round(averageAccuracy,2))
        row += 1
workbook.close()
