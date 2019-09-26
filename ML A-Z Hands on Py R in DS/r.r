# Pre-Process ----------------------------------------------------------------------
# Importing dataset
data = read.csv('data.csv')
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
split = sample.split(data$goal, SplitRatio = 0.8)
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

# Visualising the Regression Model results (for higher resolution and smoother curve)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
# normal visuaizin with x_grid instead of x

# SVR
library(e1071)
regressor=svm(formula=Salary~. ,
              data=dataset ,type='eps-regression' )#type=eps -> for regression
