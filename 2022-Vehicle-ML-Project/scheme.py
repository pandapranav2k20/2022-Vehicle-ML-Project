"""
@author: pranavveerubhotla


>>> The purpose of this project is to determine whether key environmental data 
regarding vehicles can be predicted using readily available consumer 
information. Specifically, this report aims to use Machine Learning (ML) 
techniques to make predictions regarding a vehicles combined fuel consumption 
and CO2 emissions using regression techniques.

>>> Both Multiple Linear Regression (MLR) and Artificial Neural Networks (ANNs) 
will be tested to determine which ML technique best suits the data used when 
predicting the vehicle statistics pertaining to environmental data.

"""

import numpy as np
import pandas as pd
import csv
from kmeans import kmeans
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor

df2 = pd.read_csv("MY2022 Fuel Consumption Ratings.csv")
print('ORIGINAL; DATAFRAME TYPES...\n',df2.dtypes, '\n')

my_list = []
my_values = []
for col in ['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']:
    df2[col] = df2[col].astype('category')
    my_list.append(df2[col].cat.codes)
    my_values.append(df2[col].tolist())
print('After Encoding Categogorical Data; DATAFRAME TYPES...\n',df2.dtypes, '\n')
cat_columns = df2.select_dtypes(['category']).columns
df2[cat_columns] = df2[cat_columns].apply(lambda x: x.cat.codes)


#Organize the dataframe (seperate features & labels)
feature_names = ['Model Year', 'Make', 'Model',
                 'Vehicle Class', 'Engine Size(L)', 'Cylinders', 'Transmission',
                 'Fuel Type']
features = df2[feature_names]

results = []
for i in my_values:
    results.append(list({value:"" for value in i}))

fnames = ['make_enc.csv', 'model_enc.csv', 'vc_enc.csv','trans_enc.csv', 'fuel_enc.csv']

print('We ~ENCODE~ using the following scheme... \n')
for i in range(0, len(results)):
    print('\n~', cat_columns[i])
    print('++++++++++++++++++++')
    file = open(fnames[i], 'w')
    writer = csv.writer(file)
    k=0
    for j in results[i]:
       # print('\t', j, '\t\t\t\t\t', k)
       # PRINT & also write to appropriate csv
       print(f"{'Entity: ' + j:<70} Encoded #: {k}")
       writer.writerow([j, k])
       k+=1
    file.close()



    
X = features
y1 = df2['Fuel Consumption (City (L/100 km)']
y2 = df2['Fuel Consumption(Hwy (L/100 km))']
y3 = df2['Fuel Consumption(Comb (L/100 km))']
y4 = df2['Fuel Consumption(Comb (mpg))']
y5 = df2['CO2 Emissions(g/km)']
y6 = df2['CO2 Rating']
y7 = df2['Smog Rating']

all_handles = ['Model Year', 'Make', 'Model',
                 'Vehicle Class', 'Engine Size(L)', 'Cylinders', 'Transmission',
                 'Fuel Type', 'Fuel Consumption (City (L/100 km)', 'Fuel Consumption(Hwy (L/100 km))',
                 'Fuel Consumption(Comb (L/100 km))', 'Fuel Consumption(Comb (mpg))', 'CO2 Emissions(g/km)',
                 'CO2 Rating', 'Smog Rating']
#Split the data
X1_train, X1_test, y1_train, y1_test = train_test_split(X, 
                                                    y1, 
                                                    train_size=0.8, 
                                                    random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X, 
                                                    y2, 
                                                    train_size=0.8, 
                                                    random_state=42)
X3_train, X3_test, y3_train, y3_test = train_test_split(X, 
                                                    y3, 
                                                    train_size=0.8, 
                                                    random_state=42)
X4_train, X4_test, y4_train, y4_test = train_test_split(X, 
                                                    y4, 
                                                    train_size=0.8, 
                                                    random_state=42)
X5_train, X5_test, y5_train, y5_test = train_test_split(X, 
                                                    y5, 
                                                    train_size=0.8, 
                                                    random_state=42)
X6_train, X6_test, y6_train, y6_test = train_test_split(X, 
                                                    y6, 
                                                    train_size=0.8, 
                                                    random_state=42)
X7_train, X7_test, y7_train, y7_test = train_test_split(X, 
                                                    y7, 
                                                    train_size=0.8, 
                                                    random_state=42)


'''Data ANALYSIS'''
'''Transform Categorical Data'''
types_of_vehicles = []
for i in df2['Vehicle Class']:
    types_of_vehicles.append(i)
unique_types_of_vehicles = set(types_of_vehicles)

transmissions_of_vehicles = []
for i in df2['Transmission']:
    transmissions_of_vehicles.append(i)
unique_transmissions_of_vehicles = set(transmissions_of_vehicles)

cylinders_of_vehicles = []
for i in df2['Cylinders']:
    cylinders_of_vehicles.append(i)
unique_cylinders_of_vehicles = set(cylinders_of_vehicles)

fuel_types_of_vehicles = []
for i in df2['Fuel Type']:
    fuel_types_of_vehicles.append(i)
unique_fuel_types_of_vehicles = set(fuel_types_of_vehicles)

#Plot quantities
grid_cleansed = df2.hist(xlabelsize = 5, grid = False, xrot = 45)
plt.savefig('grid.jpeg')
#grid_cleansed_stacked = df2.plot.hist(stacked = True, bins = 1000)
color = {
    "boxes": "DarkGreen",
    "whiskers": "DarkOrange",
    "medians": "DarkBlue",
    "caps": "Gray",
}

bp = df2.plot.box(color = color, sym = 'r+', vert = False, rot=45)
plt.savefig('bp.jpeg')
correlation_df = df2.corr()
correlation_df.plot(title='Correlation Plot of ALL Features & Response Vars')
plt.savefig('corr.jpeg')

pscatter_matrix = scatter_matrix(df2, alpha = 0.5, figsize = (7,9))

axes_sm = pd.plotting.scatter_matrix(df2, alpha=0.2)
for ax in axes_sm.flatten():
    ax.xaxis.label.set_rotation(90)
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_ha('right')

plt.tight_layout()
plt.gcf().subplots_adjust(wspace=0, hspace=0)
plt.show()
plt.savefig('sm.jpeg')

print('______________________________________________________________________________________')
print('______________________________________________________________________________________')
print('                          Fuel Econonomy (Comb [mpg]) Predictions                     ')
print('______________________________________________________________________________________')
print('\n ~MULTIPLE LINEAR REGRESSION')
'''
MULTIPLE
LINEAR 
REGRESSION
'''

'''Preform Multiple Linear Regression for Fuel Economy in GENERAL'''
'''USING ALL FEATURES'''
linear_modeld = LinearRegression()
linear_modeld.fit(X4_train,y4_train)
y4_test_pred = linear_modeld.predict(X4_test)
mse_all = mean_squared_error(y4_test, y4_test_pred)
r2_all = r2_score(y4_test, y4_test_pred)
print('\n[MLR-ALL Features], MSE on the test set considering ALL FEATURES is', mse_all)
print('[MLR-ALL Features], R2 on the test set considering ALL FEATURES is', r2_all)




'''Multiple Linear Rregression -- more specific -- vehicle type and transmission against y1'''
'''USING VEHICLE CLASS & CYLINDERS'''
X = df2[['Vehicle Class', 'Cylinders']].values.reshape(-1,2)
Y = y4

x = X[:, 0]
y = X[:, 1]
z = Y

x_pred = np.linspace(0, 13, 13)   # range of Vehicle Class indexes [formerly CATEGORICAL]
y_pred = np.linspace(0, 16, 16)  # range of Cylinders
xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T

mlr = linear_model.LinearRegression()
model = mlr.fit(X, Y)
predicted = model.predict(model_viz)

r2 = model.score(X, Y)

print('\n[MLR-VC & Cylinders], R2 value considering VC & Cylinders is', r2)

plt.style.use('default')

fig = plt.figure(figsize=(12, 4))

ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

axes = [ax1, ax2, ax3]

for ax in axes:
    ax.plot(x, y, z, color='purple', zorder=15, linestyle='none', marker='d', alpha=0.5)
    ax.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted, facecolor=(0,0,0,0), s=20, edgecolor='red')
    ax.set_xlabel('Vehicle Class', fontsize=12)
    ax.set_ylabel('Cylinders', fontsize=12)
    ax.set_zlabel('Fuel Consumption (Comb (mpg))', fontsize=12)
    ax.locator_params(nbins=4, axis='x')
    ax.locator_params(nbins=5, axis='x')



ax1.view_init(elev=28, azim=120)
ax2.view_init(elev=4, azim=114)
ax3.view_init(elev=60, azim=165)

fig.suptitle('Combined Fuel Economy [mpg] Predictions using MLR; '+'$R^2 = %.4f$' % r2, fontsize=20)

fig.tight_layout()
plt.savefig('Combined Fuel Economy.jpeg')

'''
ARTIFICIAL
NUERAL
NETWORKS
'''
print('\n ~ARTIFICIAL NUERAL NETWRORKS')
'''We are then going to use an ANN of 100 nodes & 1 hidden layer -- LOGISTIC'''
scaler = StandardScaler().fit(features)

ann_model1 = MLPRegressor((100), 
                      activation='logistic', 
                      solver='adam', 
                      alpha=0.0001, 
                      learning_rate='constant', 
                      learning_rate_init=0.001,
                      max_iter=50000,
                      random_state=42,
                      verbose=False)

ann_model1.fit( scaler.transform(X4_train), y4_train)

mse_train_ann1 = mean_squared_error(y4_train, ann_model1.predict(scaler.transform(X4_train)))

Y_pred_ann1 = ann_model1.predict(scaler.transform(X4_test))
mse_test_ann1 = mean_squared_error(y4_test, Y_pred_ann1)

print('[ ANN1 : 1HL-100N {LOGISTIC}] Training error (MSE): ', mse_train_ann1)
print('[ ANN1 : 1HL-100N {LOGISTIC}] Testing error (MSE):  ', mse_test_ann1)

'''We are then going to use an ANN of 20 nodes & 1 hidden layer'''
scaler = StandardScaler().fit(features)

ann_model2 = MLPRegressor((20), 
                      activation='logistic', 
                      solver='adam', 
                      alpha=0.0001, 
                      learning_rate='constant', 
                      learning_rate_init=0.001,
                      max_iter=50000,
                      random_state=42,
                      verbose=False)

ann_model2.fit( scaler.transform(X4_train), y4_train)

mse_train_ann2 = mean_squared_error(y4_train, ann_model2.predict(scaler.transform(X4_train)))

Y_pred_ann2 = ann_model2.predict(scaler.transform(X4_test))
mse_test_ann2 = mean_squared_error(y4_test, Y_pred_ann2)

print('[ ANN2 : 1HL-20N {LOGISTIC}] Training error (MSE): ', mse_train_ann2)
print('[ ANN2 : 1HL-20N {LOGISTIC}]Testing error (MSE):  ', mse_test_ann2)

'''We are then going to use an ANN of 100 nodes & 4 hidden layer'''
scaler = StandardScaler().fit(features)

ann_model3 = MLPRegressor((25, 25, 25, 25), 
                      activation='logistic', 
                      solver='adam', 
                      alpha=0.0001, 
                      learning_rate='constant', 
                      learning_rate_init=0.001,
                      max_iter=50000,
                      random_state=42,
                      verbose=False)

ann_model3.fit( scaler.transform(X4_train), y4_train)

mse_train_ann3 = mean_squared_error(y4_train, ann_model3.predict(scaler.transform(X4_train)))

Y_pred_ann3 = ann_model3.predict(scaler.transform(X4_test))
mse_test_ann3 = mean_squared_error(y4_test, Y_pred_ann3)

print('[ ANN3 : 4HL-25N/E {LOGISTIC}] Training error (MSE): ', mse_train_ann3)
print('[ ANN3 : 4HL-25N/E {LOGISTIC}]Testing error (MSE):  ', mse_test_ann3)

'''We are then going to use an ANN of 100 nodes & 1 hidden layer -- TANH'''
scaler = StandardScaler().fit(features)

ann_model4 = MLPRegressor((100), 
                      activation='tanh', 
                      solver='adam', 
                      alpha=0.0001, 
                      learning_rate='constant', 
                      learning_rate_init=0.001,
                      max_iter=50000,
                      random_state=42,
                      verbose=False)

ann_model4.fit( scaler.transform(X4_train), y4_train)

mse_train_ann4 = mean_squared_error(y4_train, ann_model4.predict(scaler.transform(X4_train)))

Y_pred_ann4 = ann_model4.predict(scaler.transform(X4_test))
mse_test_ann4 = mean_squared_error(y4_test, Y_pred_ann1)

print('[ ANN4 : 1HL-100N {TANH}] Training error (MSE): ', mse_train_ann4)
print('[ ANN4 : 1HL-100N {TANH}] Testing error (MSE):  ', mse_test_ann4)

'''We are then going to use an ANN of 100 nodes & 4 hidden layer -- TANH'''
scaler = StandardScaler().fit(features)

ann_model5 = MLPRegressor((25, 25, 25, 25), 
                      activation='tanh', 
                      solver='adam', 
                      alpha=0.0001, 
                      learning_rate='constant', 
                      learning_rate_init=0.001,
                      max_iter=50000,
                      random_state=42,
                      verbose=False)

ann_model5.fit( scaler.transform(X4_train), y4_train)

mse_train_ann5 = mean_squared_error(y4_train, ann_model5.predict(scaler.transform(X4_train)))

Y_pred_ann5 = ann_model5.predict(scaler.transform(X4_test))
mse_test_ann5 = mean_squared_error(y4_test, Y_pred_ann5)

print('[ ANN5 : 4HL-25N/E {TANH}] Training error (MSE): ', mse_train_ann5)
print('[ ANN5 : 4HL-25N/E {TANH}]Testing error (MSE):  ', mse_test_ann5)


print('______________________________________________________________________________________')
print('______________________________________________________________________________________')
print('                              CO2 Emissions [g/km]  Predictions                       ')
print('______________________________________________________________________________________')
print('\n ~MULTIPLE LINEAR REGRESSION')
'''
MULTIPLE
LINEAR 
REGRESSION
'''

'''Preform Multiple Linear Regression for Fuel Economy in GENERAL'''
'''USING ALL FEATURES'''
scaler = StandardScaler().fit(features)
linear_modeld = LinearRegression()
linear_modeld.fit(scaler.transform(X5_train),y5_train)
y5_test_pred = linear_modeld.predict(scaler.transform(X5_test))
mse_all = mean_squared_error(y5_test, y5_test_pred)
r2_all = r2_score(y5_test, y5_test_pred)
print('\n[MLR-ALL Features], MSE on the test set considering ALL FEATURES is', mse_all)
print('[MLR-ALL Features], R2 on the test set considering ALL FEATURES is', r2_all)




'''Multiple Linear Rregression -- more specific -- vehicle type and transmission against y1'''
'''USING VEHICLE CLASS & CYLINDERS'''
X = df2[['Vehicle Class', 'Cylinders']].values.reshape(-1,2)
Y = y5

x = X[:, 0]
y = X[:, 1]
z = Y

x_pred = np.linspace(0, 13, 13)   # range of Vehicle Class indexes [formerly CATEGORICAL]
y_pred = np.linspace(0, 16, 16)  # range of Cylinders
xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T

mlr = linear_model.LinearRegression()
model = mlr.fit(X, Y)
predicted = model.predict(model_viz)

r2 = model.score(X, Y)

print('\n[MLR-VC & Cylinders], R2 value considering VC & Cylinders is', r2)

plt.style.use('default')

fig = plt.figure(figsize=(12, 4))

ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

axes = [ax1, ax2, ax3]

for ax in axes:
    ax.plot(x, y, z, color='red', zorder=15, linestyle='none', marker='s', alpha=0.5)
    ax.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted, facecolor=(0,0,0,0), s=20, edgecolor='blue')
    ax.set_xlabel('Vehicle Class', fontsize=12)
    ax.set_ylabel('Cylinders', fontsize=12)
    ax.set_zlabel('Carbon Emissions [g/km]', fontsize=12)
    ax.locator_params(nbins=4, axis='x')
    ax.locator_params(nbins=5, axis='x')



ax1.view_init(elev=28, azim=120)
ax2.view_init(elev=4, azim=114)
ax3.view_init(elev=60, azim=165)

fig.suptitle('Carbon Emissions [g/km] Predictions using MLR; '+'$R^2 = %.4f$' % r2, fontsize=20)

fig.tight_layout()
plt.savefig('Carbon Emissions.jpeg')

'''
ARTIFICIAL
NUERAL
NETWORKS
'''
print('\n ~ARTIFICIAL NUERAL NETWRORKS')
'''We are then going to use an ANN of 100 nodes & 1 hidden layer'''
scaler = StandardScaler().fit(features)

ann_model1 = MLPRegressor((100), 
                      activation='logistic', 
                      solver='adam', 
                      alpha=0.0001, 
                      learning_rate='constant', 
                      learning_rate_init=0.001,
                      max_iter=50000,
                      random_state=42,
                      verbose=False)

ann_model1.fit( scaler.transform(X5_train), y5_train)

mse_train_ann1 = mean_squared_error(y5_train, ann_model1.predict(scaler.transform(X5_train)))

Y_pred_ann1 = ann_model1.predict(scaler.transform(X5_test))
mse_test_ann1 = mean_squared_error(y5_test, Y_pred_ann1)

print('[ ANN1 : 1HL-100N {LOGISTIC}] Training error (MSE): ', mse_train_ann1)
print('[ ANN1 : 1HL-100N {LOGISTIC}] Testing error (MSE):  ', mse_test_ann1)

'''We are then going to use an ANN of 20 nodes & 1 hidden layer'''
scaler = StandardScaler().fit(features)

ann_model2 = MLPRegressor((20), 
                      activation='logistic', 
                      solver='adam', 
                      alpha=0.0001, 
                      learning_rate='constant', 
                      learning_rate_init=0.001,
                      max_iter=50000,
                      random_state=42,
                      verbose=False)

ann_model2.fit( scaler.transform(X5_train), y5_train)

mse_train_ann2 = mean_squared_error(y5_train, ann_model2.predict(scaler.transform(X5_train)))

Y_pred_ann2 = ann_model2.predict(scaler.transform(X5_test))
mse_test_ann2 = mean_squared_error(y5_test, Y_pred_ann2)

print('[ ANN2 : 1HL-20N {LOGISTIC}] Training error (MSE): ', mse_train_ann2)
print('[ ANN2 : 1HL-20N {LOGISTIC}]Testing error (MSE):  ', mse_test_ann2)

'''We are then going to use an ANN of 100 nodes & 4 hidden layer'''
scaler = StandardScaler().fit(features)

ann_model3 = MLPRegressor((25, 25, 25, 25), 
                      activation='logistic', 
                      solver='adam', 
                      alpha=0.0001, 
                      learning_rate='constant', 
                      learning_rate_init=0.001,
                      max_iter=50000,
                      random_state=42,
                      verbose=False)

ann_model3.fit( scaler.transform(X5_train), y5_train)

mse_train_ann3 = mean_squared_error(y5_train, ann_model3.predict(scaler.transform(X5_train)))

Y_pred_ann3 = ann_model3.predict(scaler.transform(X4_test))
mse_test_ann3 = mean_squared_error(y5_test, Y_pred_ann3)

print('[ ANN3 : 4HL-25N/E {LOGISTIC}] Training error (MSE): ', mse_train_ann3)
print('[ ANN3 : 4HL-25N/E {LOGISTIC}]Testing error (MSE):  ', mse_test_ann3)

'''We are then going to use an ANN of 100 nodes & 1 hidden layer -- TANH'''
scaler = StandardScaler().fit(features)

ann_model4 = MLPRegressor((100), 
                      activation='tanh', 
                      solver='adam', 
                      alpha=0.0001, 
                      learning_rate='constant', 
                      learning_rate_init=0.001,
                      max_iter=50000,
                      random_state=42,
                      verbose=False)

ann_model4.fit( scaler.transform(X5_train), y5_train)

mse_train_ann4 = mean_squared_error(y5_train, ann_model4.predict(scaler.transform(X5_train)))

Y_pred_ann4 = ann_model4.predict(scaler.transform(X5_test))
mse_test_ann4 = mean_squared_error(y5_test, Y_pred_ann4)

print('[ ANN4 : 1HL-100N {TANH}] Training error (MSE): ', mse_train_ann4)
print('[ ANN4 : 1HL-100N {TANH}] Testing error (MSE):  ', mse_test_ann4)

'''We are then going to use an ANN of 100 nodes & 4 hidden layer -- TANH'''
scaler = StandardScaler().fit(features)

ann_model5 = MLPRegressor((25, 25, 25, 25), 
                      activation='tanh', 
                      solver='adam', 
                      alpha=0.0001, 
                      learning_rate='constant', 
                      learning_rate_init=0.001,
                      max_iter=50000,
                      random_state=42,
                      verbose=False)

ann_model5.fit( scaler.transform(X5_train), y5_train)

mse_train_ann5 = mean_squared_error(y5_train, ann_model5.predict(scaler.transform(X5_train)))

Y_pred_ann5 = ann_model5.predict(scaler.transform(X5_test))
mse_test_ann5 = mean_squared_error(y5_test, Y_pred_ann5)

print('[ ANN5 : 4HL-25N/E {TANH}] Training error (MSE): ', mse_train_ann5)
print('[ ANN5 : 4HL-25N/E {TANH}]Testing error (MSE):  ', mse_test_ann5)