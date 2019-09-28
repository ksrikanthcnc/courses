# Pre-Process ----------------------------------------------------------------------
# Importing dataset
data = read.csv('data.csv') # '=' r '<-'
data_subset = data[, 2:3]

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
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
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


