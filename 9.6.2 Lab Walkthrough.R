#################################
##### 9.6.2 Lab Walkthrough #####
#################################

# Just as in the lab 9.6.1, we must load the Library "e1071", as this library contains the svm funtion we want to use
library(e1071)

# We set the seed so we can replicate our findings, then we create an x and y matrix, and then combine them to create a data frame.
# The data in our data frame in generated with non-linear class boundary.
set.seed(1)
x=matrix(rnorm (200*2) , ncol=2)
x[1:100,]=x[1:100,]+2
x[101:150 ,]=x[101:150,]-2
y=c(rep(1,150) ,rep(2,50))
dat=data.frame(x=x,y=as.factor(y))

# We plot the to ensure that the class boundary is indeed non-linear
plot(x, col=y)

# Next we radnomly split our data into training and testing groups, and fit the training data using the svm() function. 
# The difference between this svm fit and the one we did in 9.6.1 is that our kernel is now "linear" and the gamma is now 1, for this
# non-linear problem. 
train=sample (200,100)
svmfit=svm(y???., data=dat[train ,], kernel ="radial", gamma=1,cost=1)
plot(svmfit , dat[train ,])

# The plot shows that the resulting SVM has a decidedly non-linear boundary. We use the summary() function  to obtain more information 
# about the SVM fit, such as Number of Support Vectors or Number of Classes.

summary(svmfit)

# We see from the that there are is a decent number of training errors in this SVM fit. We can reduce these training errors by 
# increasing the value of cost, however, this can cause a more irregular decision boundary and rissk of overfitting the data.

svmfit=svm(y???., data=dat[train ,], kernel ="radial",gamma=1,cost=1e5)
plot(svmfit ,dat[train ,])

# We can perform cross-validation using tune() to select the best choice of gamma and cost for an SVM with a "radial" kernel

set.seed(1)
tune.out=tune(svm , y???., data=dat[train ,], kernel ="radial",ranges=list(cost=c(0.1,1,10,100,1000), gamma=c(0.5,1,2,3,4) ))
summary (tune.out)

# In doing this, we concluded that the best choice of parameters involves cost=1 and gamma=2. We
# then  view the test set predictions for this model by using the predict() function. In order to do this we have to subset our 
# dataframe using -train as an index set.

table(true=dat[-train ,"y"], pred=predict (tune.out$best.model ,newdata =dat[-train ,]))

# In conclusion we see notice that 12/100 (12%) of the test observations are misclassified by this SVM.