# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import Dataset
df = pd.read_csv('data.csv')
# : required to get a matrix(X), and should not be used for a vector(y)
X = df.iloc[:, :-1].values #can also select columns by [1,2,3]
y = df.iloc[:, 5].values

# Pre-Processing -------------------------------------------------------------------------
# Missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, :-1])
X[:, :-1] = imputer.transform(X[:, :-1])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
#Avoid Dummy Variable Trap
X = X[:, 1:]

# Splitting to train and test sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_X = sc_X.fit(X_train)
X_train = sc_X.transform(X_train)
X_test = sc_X.transform(X_test)

# Regression ------------------------------------------------------------------------------
# Simple Regressor fitting
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting
y_pred = regressor.predict(X_test)

# Visualizing
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('TITLE')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

# Multiple linear Regression
# Same as Simple Linear Regression

# Backward Elimination
# Prepending constant 1 column for backward elimination
import statsmodel.formula.api as sm
X = np.append(arr = np.ones((no_of_cols, 1)).astype(int), values = X, axis = 1)
# Repeat by deleting (one highest)column in X_opt whose P value in summary is > predefined threshold (5%)
# todo automate this
X_opt = X[: [0, 1, 2, 3, 4]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# Polynomial Linear Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
# Normal visualizing

# Fitting SVR to the dataset (needs featur scaling)
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')

y_pred = sc_y.inverse_transform(y_pred)

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)

# Classification ---------------------------------------------------------------------------
# Probably needs feature scaling
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green'))) # more colours for more classes
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0) # kernel = 'rbf',... for non-linear guassian

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

# Clustering ----------------------------------------------------------------------------------
# K-Means
from sklearn.cluster import KMeans
# To find optimal number of clusters
# todo fetch optimal clusters automatically
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)#within cluster sum of squares
plt.plot(range(1, 11), wcss)
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters (2D)
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Careful') #y_kmeans == 0, 1   =>o-index and 1 = value
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids') # only of kmeans not for HC
plt.legend()
plt.show()

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward')) #Algorithm of hirachical clustering 
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward') #distance to do the linkage 
y_hc = hc.fit_predict(X)

# Apriori ------------------------------------------------------------------------------------
# todo, inculcate apriori_module file here
from apriori_module import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)

# Reinforcement Learning ---------------------------------------------------------------------
# Upper Confidence Bound

# Random to check performance
# Implementing Random Selection
# import random
# N = 10000
# d = 10
# ads_selected = []
# total_reward = 0
# for n in range(0, N):
#     ad = random.randrange(d)
#     ads_selected.append(ad)
#     reward = dataset.values[n, ad]
#     total_reward = total_reward + reward

# Implementing UCB
import math
N = 10000 #number of rounds
d = 10 # number of adds

ads_selected = []
numbers_of_selections = [0] * d #init arr adds of size d assign 0
sums_of_rewards = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])#n + 1 =>total played ,, numbers_of_selections[i] ->number of times add got selected
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400 #1e400 used because upper_bound > max_upper_bound should be handled. see the iteration
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

# Implementing Thompson Sampling
import random
N = 10000
d = 10
ads_selected = []
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    total_reward = total_reward + reward

# Natural Language Processing ----------------------------------------------------------------
import re #cleaning library
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []

# Pre-Process
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] #set improves execution speed
    #ps object do stemming ex: love ,loving etc to love
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model / metrix with lot of 0's called sparse matrix
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) #max_features :removes unrelavent words like names etc 
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Split into training, test sets
# Naive Bayes,  Decision Tree Classification, Random Forest Classification, 
# Making the Confusion Matrix
accuracy=(cm.item(0,0)+cm.item(1,1))/len(X_test)

# Deep Learning ------------------------------------------------------------------------------
# Artificial Neural Networks
# 'Theano', 'TensorFlow' (mostly to develop your own) for efficient utiliation of CPU, GPU (wrap numpy)
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential# required to init ANN
from keras.layers import Dense #required to build layers of ANN

classifier = Sequential()

# Adding the input layer and the first hidden layer
# output dim=>average #output+input=> (no of input columns+ output )/2=>(11+1)/2
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))#
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
#if output is more than 1 catagorical variable apply activation='softmax'and output_dim="#outputs"
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#adam-type of schocastic gradient decent   #loss = 'binary_crossentropy'
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Convolutional Neural Network
# Part 1 - Building the CNN
# Importing the Keras libraries and packages
from keras.models import Sequential # initialize nural network
from keras.layers import Conv2D # add convolution layers 2D:to images
from keras.layers import MaxPooling2D # do pooling
from keras.layers import Flatten #flattning.
from keras.layers import Dense # add fully connected layer to ANN
# Initialising the CNN
classifier = Sequential()
classifier.add(Conv2D(32,( 3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2))) #pool size =>2x2 
# Adding second convolution layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Pre-Processing (Image Augmentation) to generate modified images from small dataset to have more images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(   training_set,
                            steps_per_epoch=8000,
                            epochs=25,
                            validation_data=test_set,
                            validation_steps=2000)

# Testing
from keras.preprocessing import image as image_utils
import numpy as np
 
test_image = image_utils.load_img('dataset/test_set/cats/cat.4001.jpg', target_size=(64, 64))
test_image = image_utils.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = classifier.predict_on_batch(test_image)
if(result==1):
    print("dog")
else:
    print("cat")
    
from skimage.io import imread
from skimage.transform import resize
img = imread('dataset/test_set/dogs/dog.4001.jpg') #make sure that path_to_file contains the path to the image you want to predict on. 
img = resize(img, (64, 64))
img = np.reshape(img,(1,64,64,3))
img = img/(255.0)
prediction = classifier.predict_classes(img)
print (prediction)    

# Applying Dimensionality Reduction ----------------------------------------------------------
# PCA (unsupervised)
# Needs Feature Scaling
from sklearn.decomposition import PCA
pca = PCA(n_components = 2) # '...' = None for all, to know the optimal number
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

# LDA (supervised)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

# Kernel PCA (non-linearly seperable); maps to higher dimensionality, applies PCA, and finally makes it linarly seperable
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 2, kernel = 'rbf')
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)

# Model selection ------------------------------------------------------------------------
# K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

# Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
# example of SVM ie classifier is SVM
parameters = [{ 'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              { 'C': [1, 10, 100, 1000], 'kernel': ['rbf'],
                'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

# XG Boost-------------------------------------------------------------------------------------
# No need to feature scale
from xgboost import XGBClassifier
classifier = XGBClassifier()

