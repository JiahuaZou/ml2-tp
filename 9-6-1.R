# Chapter 9 Lab: Support Vector Machines

#################################
##### 9.6.1 Lab Walkthrough #####
#################################

# Support Vector Classifier

# Generate some data by creating a matrix, x, with 20 observations in 2 classes
# And we make the y variable, including 1 and -1, and 10 in each class
# And then we set the mean to 1 for each cordinates
set.seed(1)
x=matrix(rnorm(20*2), ncol=2)
y=c(rep(-1,10), rep(1,10))
x[y==1,]=x[y==1,] + 1
plot(x, col=(3-y))

# Next we make a dataframe and turn y into factor variable
# Then we call the e1071 library, which contains implementations 
# for a number of statistical learning methods. 
# In particular, the svm() function can be used to fit a
# support vector classifier when the argument kernel="linear" is used.
dat=data.frame(x=x, y=as.factor(y))
library(e1071)

# Next, we fit the support vector classifier. Note that in order for the svm() function to 
# perform classification, we must encode the response as a factor variable.
svmfit=svm(y~., data=dat, kernel="linear", cost=10,scale=FALSE)
plot(svmfit, dat)

# The support points can be found by using the $index in the svmfit
# Keep in mind that the the support vectors are the points that are on the boundary, 
# or on the wrong side of the boundary.
svmfit$index
# We call the summary function, and then we get the number of support vectors
summary(svmfit)

svmfit=svm(y~., data=dat, kernel="linear", cost=0.1,scale=FALSE)
plot(svmfit, dat)
svmfit$index

# Then we apply the tune() function to perform ten-kold cross-validation on a set of models of interest.
set.seed(1)
tune.out=tune(svm,y~.,data=dat,kernel="linear",ranges=list(cost=c(0.001, 0.01, 0.1, 1,5,10,100)))
# We can easily access the cross-validation errrors for each of these models using the summary() command.
summary(tune.out)
# And we can access the best model by using $best.model
bestmod=tune.out$best.model
summary(bestmod)

# Then we generate a test data set using the same method as above,
# and then we can predict the class label on a set of test observations, at any given value of the cost parameter.
# Here we use the best model obtained through cross-validation to make predictions.
xtest=matrix(rnorm(20*2), ncol=2)
ytest=sample(c(-1,1), 20, rep=TRUE)
xtest[ytest==1,]=xtest[ytest==1,] + 1
testdat=data.frame(x=xtest, y=as.factor(ytest))
ypred=predict(bestmod,testdat)
table(predict=ypred, truth=testdat$y)

# What if we use cost = 0.01 instead? Let's try:
svmfit=svm(y~., data=dat, kernel="linear", cost=.01,scale=FALSE)
ypred=predict(svmfit,testdat)
table(predict=ypred, truth=testdat$y)

# Now consider a situation in which the two classes are linearly separable.
x[y==1,]=x[y==1,]+0.5
plot(x, col=(y+5)/2, pch=19)

# According to the plot, the observations are just barely linearly separable. 
# We fit the support vector classifier and plot the resulting hyperplane,
# using a very large value of cost so that no observations are misclassified.
dat=data.frame(x=x,y=as.factor(y))
svmfit=svm(y~., data=dat, kernel="linear", cost=1e5)
summary(svmfit)
plot(svmfit, dat)
# In the summary, no training errors were made, and only 3 support vectors were used,
# in this case, we can see from the figure that the margin is very narrow;
# So it seems likely that this model will perfom poorly on the test data.
# How about let's try with a smaller value of cost = 1.
svmfit=svm(y~., data=dat, kernel="linear", cost=1)
summary(svmfit)
plot(svmfit,dat)
# From this plot, we can see that we misclassify one training observation,
# but we also obtain a much wider margin and make use of five support vectors.
# And it seems likely that this model will perform better on test data than the previous model.
