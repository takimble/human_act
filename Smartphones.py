
import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt 
import seaborn as sb
# matplotlib inline


#smart phone data
test = pd.read_csv("https://s3.amazonaws.com/hackerday.datascience/112/train.csv")
train = pd.read_csv("https://s3.amazonaws.com/hackerday.datascience/112/train.csv")

# number of records and column headers
# print('Train Data', train.shape, '\n', train.columns)
# print('\nTest Data', test.shape)

# activity is the target variable 
# print('Train labels', train['Activity'].unique(), '\nTest Labels', test['Activity'].unique())

# cross tab for identifying 
# x = subject , y = activity
# pd.crosstab(train.subject, train.Activity)


# subject 1 data and activities 
sub15 = train.loc[train['subject']==1]
#distribution by subject
sub15.head()

train.head()
# count of activities 
# print(train.subject.value_counts())
# print(train.Activity.value_counts())
# print(test.Activity.value_counts())

# graphical representation
fig = plt.figure(figsize=(32,24))
ax1 = fig.add_subplot(221)
ax1= sb.stripplot(x='Activity', y=sub15.iloc[:,0], data=sub15, jitter=True)
ax2=fig.add_subplot(222)
ax2=sb.stripplot('Activity', y=sub15.iloc[:,1], data=sub15, jitter=True )
# plt.show()

# shuffle data to change the order sequence 
from sklearn.utils import shuffle 

test = shuffle(test)
train = shuffle(train)

# seperating data inputs and output lables
trainData = train.drop('Activity', axis=1).values
trainLabel = train.Activity.values

testData = test.drop('Activity', axis=1).values
testLabel = test.Activity.values


# encoding labels
from sklearn import preprocessing

encoder = preprocessing.LabelEncoder()

# encoding test labels 
encoder.fit(testLabel)
testLabelE = encoder.transform(testLabel)

# endcoding train labels
encoder.fit(trainLabel)
trainLabelE = encoder.transform(trainLabel)


# target variable is catergorical 
# independant variables are numbers (classification case)

#classification models:
#decision treee
#SVM
#Neural network
#RF
#GBM
#DNN

#applying supervised neural network using multi layer preception
import sklearn.neural_network as nn 

#very basic configurtaion 
mlpSGD = nn.MLPClassifier(hidden_layer_sizes=(90,),max_iter=1000 , alpha=0.0001, solver = 'sgd' , verbose=10, tol= 1e-19 ,random_state =1, learning_rate_init=.001)

mlpADAM = nn.MLPClassifier (hidden_layer_sizes=(90,), max_iter=1000, alpha=0.0001, solver='adam', verbose=10, tol=1e-19, random_state=1, learning_rate_init=.001)        

mlpLBFGS = nn.MLPClassifier(hidden_layer_sizes=(90,), max_iter=1000, alpha=0.0001, solver='lbfgs', verbose=10, tol=1e-19, random_state=1, learning_rate_init=.001)

# nnModelSGD = mlpSGD.fit(trainData, trainLabelE)
# nnModelLSGD = mlpLBFGS.fit(trainData, trainLabelE)
# nnModelADAM = mlpADAM.fit(trainData, trainLabelE)


test_df = pd.read_csv("https://s3.amazonaws.com/hackerday.datascience/112/train.csv")
train_df = pd.read_csv("https://s3.amazonaws.com/hackerday.datascience/112/train.csv")

unique_activities = train_df.Activity.unique()
#print('Number of unique activities: {}'.format(len(unique_activities)))

replacer = {}
for i, activity in enumerate(unique_activities):
    replacer[activity] = i
train_df.Activity = train_df.Activity.replace(replacer)
test_df.Activity = test_df.Activity.replace(replacer)
train_df.head(10)

train_df = train_df.drop('subject', axis=1)
test_df = test_df.drop('subject', axis=1)

def get_all_data():
    train_values = train_df.values
    test_values = test_df.values
    np.random.shuffle(train_values)
    np.random.shuffle(test_values)
    x_train = train_values[:, :-1]
    x_test = test_values[:, :-1]
    y_train = train_values[:,-1]
    y_test = test_values[:,-1]
    return x_train, x_test, y_train, y_test

from sklearn.linear_model import LogisticRegression
x_train, x_test, y_train, y_test = get_all_data()

model = LogisticRegression(C=10,multi_class='ovr', solver= 'liblinear', n_jobs=1 )
model.fit(x_train, y_train)
model.score(x_test, y_test)
    #logical regression : 87%


# transformations
from sklearn.decomposition import PCA
x_train, x_test, y_train, y_test = get_all_data() #generate training set
pca = PCA(n_components=200) # init PCA
pca.fit(x_train) # applying PCA
x_train = pca.transform(x_train)# transforming the dataset 
x_test = pca.transform(x_test) 

model.fit(x_train,y_train)# creating model
model.score(x_test, y_test)# Worse performance but trains faster


# scale feature to be between -1 and 1
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x_train, x_test, y_train, y_test = get_all_data()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model.fit(x_train, y_train)
model.score(x_test, y_test)
# Better performance


from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils.np_utils import to_categorical

x_train, x_test, y_train, y_test = get_all_data()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_input = x_train.shape[1] # number of features 
n_output = 6 # numner of possible labels
n_samples = x_train.shape[0] #number of training samples
n_hidden_units = 60
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(Y_train.shape), print(Y_test.shape)

def create_model():
    model = Sequential()
    model.add(Dense(n_hidden_units,input_dim=n_input, activation='relu'))
    model.add(Dense(n_hidden_units,input_dim=n_input, activation='relu'))
    model.add(Dense(n_output, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics= ['accuracy'])
    return model

estimator = KerasClassifier(build_fn=create_model, epochs=20, batch_size=10, verbose=False)
estimator.fit(x_train,y_train)
print('Score:{}'.format(estimator.score(x_test,y_test)))