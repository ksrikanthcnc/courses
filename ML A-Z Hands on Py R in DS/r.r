# Pre-Process ----------------------------------------------------------------------
# Importing dataset
data = read.csv('data.csv') # '=' r '<-'; 'header = TRUE/FALSE'
data_subset = data[, 2:3]
# To re-arrange columns
# data = data[c(2, 3, 1)]

# Missing data
data$colName = ifelse(is.na(data$colName),
                        ave(data$colName, FUN = function(x) mean(x, na.rm = TRUE)),
                      data$colName)

# Encoding categorical data
data$colName = factor(data$colName,
                      levels = c('cat1', 'cat2'),
                      labels = c(1,2))

# Splitting into training and testing
library(caTools)
set.seed(123)
split = sample.split(data$goal, SplitRatio = 0.8) #SplitRation = % in training
training_set = subset(data, split == TRUE)
test_set = subset(data, split == FALSE)

# Feature Scaling (scaling cols 2,3)
training_set[, 2:3] = scale(training_set[, 2:3])
test_set[, 2:3] = scale(test_set[, 2:3])

# Regression ------------------------------------------------------------------------
# Simple linear
regressor = lm(formula = goal ~ col,
               data = training_set)
summary(regressor)

y_pred = predict(regressor, newdata = test_set)
# Single data point
y_pred = predict(regressor, data.frame(Level = 6.5))

# Visualising
library(ggplot2)
ggplot() + 
  geom_point(aes(x = training_set$col, y = training_set$goal),
             color = 'red') +
  geom_line(aes(x = training_set$col, y = predict(regressor, newdata = training_set)),
                color = 'blue') + 
  ggtitle('TITLE') +
  xlab('X-axis') +
  ylab('Y-axis')

# Multiple Regressor
regressor = lm(formula = goal ~ col1 + col2 + col3,
# regressor = lm(formula = goal ~ ., ( '.' can be used for all other cols)
               data = training_set)

# Backward elimination
# Repeat by deleting (one highest)column whose P value in summary is > predefined threshold (5%)
regressor = lm(formula = goal ~ col1 + col2 + col3,
               data = training_set)
summary(regressor)

# Polynomial Regression
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~ .,
              data = dataset)

# Visualising the poly Regression Model results (for higher resolution and smoother curve)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
# normal visuaizin with x_grid instead of x in geom_line

# SVR
library(e1071)
regressor=svm(formula=Salary~. ,
              data=dataset,
              type='eps-regression',
              kernel = 'radial')

# Decision Tree Regression
library(rpart)
regressor = rpart(formula = Salary ~ .,
                  data = dataset,
                  control = rpart.control(minsplit = 1))
# Plotting the tree
plot(regressor)
text(regressor)

library(randomForest)
set.seed(1234)
regressor = randomForest(x = dataset[1], #using [ ] gives a data frame
                         y = dataset$Salary, #using $ sign gives vector
                         ntree = 20000)

# Classification ----------------------------------------------------------------------------
# Probably needs feature scaling
# Logistic Regression
classifier = glm(formula = Purchased ~ .,
                 family = binomial, #for logistic regression
                 data = training_set)

# Predicting the Test set results
prob_pred = predict(classifier, type = 'response', newdata = test_set[-3])
y_pred = ifelse(prob_pred > 0.5, 1, 0)

# Making the Confusion Matrix
cm = table(test_set[, 3], y_pred > 0.5)

# Visualising the Training set results
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Logistic Regression (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato')) # mre colours for more classes
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# Fitting K-NN to the Training set and Predicting the Test set results
library(class)
y_pred = knn(train = training_set[, -3],
             test = test_set[, -3],
             cl = training_set[, 3],
             k = 5,
             prob = TRUE)
# to plot, use y_grid = knn(...) model

# Fitting SVM to the Training set
library(e1071)
classifier = svm(formula = Purchased ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'linear')  # kernel = 'radial',... for non-linear guassian

# Naive Bayes
# Might need encoding(factor-ing), doesn't by default
library(e1071)
classifier = naiveBayes(x = training_set[-3],
                        y = training_set$Purchased)

# Decision Tree
library(rpart)
classifier = rpart(formula = Purchased ~ .,
                   data = training_set)
y_pred = predict(classifier, newdata = test_set[-3], type = 'class') # 'type' parameter to directly get class
plot(classifier)
text(classifier)

# Random Forest
library(randomForest)
classifier = randomForest(x = training_set[-3],
                          y = training_set$Purchased,
                          ntree = 500)

# Clustering ----------------------------------------------------------------------------------
# K-Means
wcss = vector()
for (i in 1:10) wcss[i] = sum(kmeans(dataset, i)$withinss)
plot(1:10,
     wcss,
     type = 'b',
     main = paste('The Elbow Method'),
     xlab = 'Number of clusters',
     ylab = 'WCSS')

# Fitting K-Means to the dataset
kmeans = kmeans(x = dataset, centers = 5)
y_kmeans = kmeans$cluster

# Visualising the clusters
library(cluster)
clusplot(dataset,
         y_kmeans,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Clusters of customers'),
         xlab = 'Annual Income',
         ylab = 'Spending Score')

# Using the dendrogram to find the optimal number of clusters
dendrogram = hclust(d = dist(dataset, method = 'euclidean'), method = 'ward.D')
plot(dendrogram,
     main = paste('Dendrogram'),
     xlab = 'Customers',
     ylab = 'Euclidean distances')

# Fitting Hierarchical Clustering to the dataset
hc = hclust(d = dist(dataset, method = 'euclidean'), method = 'ward.D')
y_hc = cutree(hc, 5)

# Apriori ------------------------------------------------------------------------------------
# Create a sparse matrix suitable for algo
library(arules)
dataset = read.transactions('data.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10) # to choose thresholds (support, confidence)

# Apriori
# trial and error for (support,) confidence
rules = apriori(data = dataset, parameter = list(support = 0.003, confidence = 0.8))
inspect(sort(rules, by = 'lift')[1:10])

# Eclat
rules = eclat(data = dataset, parameter = list(support = 0.003, minlen = 2))
inspect(sort(rules, by = 'support')[1:10])

# Reinforcement Learning ---------------------------------------------------------------------
# Upper Confidence Bound

# Random to check performance
# Implementing Random Selection
# N = 10000
# d = 10
# ads_selected = integer(0)
# total_reward = 0
# for (n in 1:N) {
#   ad = sample(1:10, 1)
#   ads_selected = append(ads_selected, ad)
#   reward = dataset[n, ad]
#   total_reward = total_reward + reward
# }

# Implementing UCB
N = 10000
d = 10
ads_selected = integer(0)
numbers_of_selections = integer(d)
sums_of_rewards = integer(d)
total_reward = 0
for (n in 1:N) {
  ad = 0
  max_upper_bound = 0
  for (i in 1:d) {
    if (numbers_of_selections[i] > 0) {
      average_reward = sums_of_rewards[i] / numbers_of_selections[i]
      delta_i = sqrt(3/2 * log(n) / numbers_of_selections[i])
      upper_bound = average_reward + delta_i
    } else {
        upper_bound = 1e400
    }
    if (upper_bound > max_upper_bound) {
      max_upper_bound = upper_bound
      ad = i
    }
  }
  ads_selected = append(ads_selected, ad)
  numbers_of_selections[ad] = numbers_of_selections[ad] + 1
  reward = dataset[n, ad]
  sums_of_rewards[ad] = sums_of_rewards[ad] + reward
  total_reward = total_reward + reward
}

# Visualising the results
hist(ads_selected,
     col = 'blue',
     main = 'Histogram of ads selections',
     xlab = 'Ads',
     ylab = 'Number of times each ad was selected')

# Implementing Thompson Sampling
N = 10000
d = 10
ads_selected = integer(0)
numbers_of_rewards_1 = integer(d)
numbers_of_rewards_0 = integer(d)
total_reward = 0
for (n in 1:N) {
  ad = 0
  max_random = 0
  for (i in 1:d) {
    random_beta = rbeta(n = 1,
                        shape1 = numbers_of_rewards_1[i] + 1,
                        shape2 = numbers_of_rewards_0[i] + 1)
    if (random_beta > max_random) {
      max_random = random_beta
      ad = i
    }
  }
  ads_selected = append(ads_selected, ad)
  reward = dataset[n, ad]
  if (reward == 1) {
    numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
  } else {
    numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
  }
  total_reward = total_reward + reward
}

# Natural Language Processing ----------------------------------------------------------------
# Importing the dataset
dataset_original = read.delim('Restaurant_Reviews.tsv', quote = '', stringsAsFactors = FALSE)

# Cleaning the texts
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset_original$Review))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords())
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)

# Creating the Bag of Words model
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999)
dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_original$Liked

# Encoding the target feature as factor
# Splitting the dataset into the Training set and Test set
# Fitting Random Forest Classification to the Training set
# Predicting the Test set results
# Making the Confusion Matrix

# Deep Learning ------------------------------------------------------------------------------
# Artificial Neural Network

library(h2o)
# All cores
h2o.init(nthreads = -1)
# h2o.shutdown()
model = h2o.deeplearning(y = 'Exited',
                         training_frame = as.h2o(training_set),
                         activation = 'Rectifier',
                         hidden = c(5,5),
                         epochs = 100,
                         train_samples_per_iteration = -2)

# Applying Dimensionality Reduction ----------------------------------------------------------
# PCA (unsupervised)
# Needs Feature Scaling
library(caret)
library(e1071)
pca = preProcess(x = training_set[-14], method = 'pca', pcaComp = 2)
training_set = predict(pca, training_set)
test_set = predict(pca, test_set)

# LDA (supervised)
library(MASS)
lda = lda(formula = data$goal = .,
          data = training_set) # max possible newcols = classes-1
training_set = as.data.frame(predict(lda, training_set)) # convert matrix to data frame
test_set = as.data.frame(predict(lda, test_set))

# Kernel PCA (non-linearle seperable)
library(kernlab)
kpca = kpca(~., data = training_set[-3], kernel = 'rbfdot', features = 2)
training_set_pca = as.data.frame(predict(kpca, training_set))
training_set_pca$goal = training_set$goal

# Model selection ------------------------------------------------------------------------
# K-Fold Cross Validation
library(caret)
folds = createFolds(training_set$goal, k = 10)
cv = lapply(folds, function(x) {
  # Model, return accuracy
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  classifier = svm(formula = Purchased ~ .,
                   data = training_fold,
                   type = 'C-classification',
                   kernel = 'radial')
  y_pred = predict(classifier, newdata = test_fold[-3])
  cm = table(test_fold[, 3], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
})
accuracy = mean(as.numeric(cv))

# Grid Search to find the best model and the best parameters
library(caret)
# caret takes care of Grid Search
classifier = train(form = goal ~ ., data = training_set, method = 'svmRadial') #for SVM, refer caret documentation
classifier
classifier$bestTune

# XG Boost-------------------------------------------------------------------------------------
library(xgboost)
classifier = xgboost(data = as.matrix(training_set[-11]), label = training_set$goal, nrounds = 10)

