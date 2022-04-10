#Loading libraries
library(tidyverse)
library(ggplot2)
library(dplyr)
library(moments)
library(e1071)
library(caret)
library(kableExtra)
library(corrplot)
library(knitr)
library(lattice)
library(psych)
library(car)

#Loading dataset
df <- read.csv("star_classification.csv")
df
nrow(df)

#check for missing data
sum(is.na(df))

#creating predictor df and response df and dropping unwanted predictors
predictors<-df%>%select(c(-obj_ID, -class, -rerun_ID,-run_ID, -cam_col,-field_ID, -spec_obj_ID, -fiber_ID, -MJD, -plate))
response<-df%>%select(c(class))

predictors
response

filteredPredictors<-df%>%select(c(-obj_ID, -class, -rerun_ID,-run_ID, -cam_col,-field_ID, -spec_obj_ID, -fiber_ID, -MJD, -plate))

#checking correlation
correlations = cor(predictors)
corrplot(correlations, order = "hclust", method = "number", type = "lower")

highCorr <- findCorrelation(correlations, cutoff = .95)
length(highCorr)

##filteredPredictors <- predictors[, -highCorr]

#dim(predictors)
#dim(filteredPredictors)

##filteredCorrelations = cor(filteredPredictors)
##corrplot(filteredCorrelations, order = "hclust")

#plotting histograms to see distribution
filteredPredictors %>%
  gather() %>% 
  ggplot(aes(x = value))+
  geom_histogram() +
  facet_wrap(~ key, scales = "free") +
  labs(x = NULL, y = NULL) +
  theme_bw() +
  theme(axis.ticks.y=element_blank())

#checking skewness values
skewValues <- apply(filteredPredictors, 2, skewness)
skewValues

#determining bad data
describe(filteredPredictors$u)
describe(filteredPredictors$g)
describe(filteredPredictors$z)

filteredPredictors[which(filteredPredictors$u <0), ]
filteredPredictors[which(filteredPredictors$g <0), ]
filteredPredictors[which(filteredPredictors$z <0), ]

#dropping bad observation
filteredPredictors<-filteredPredictors%>%slice(-c(79544))
response<-response%>%slice(-c(79544))

#checking skewness values
skewValues <- apply(filteredPredictors, 2, skewness)
skewValues

#pre-processing transformations
trans <- preProcess(filteredPredictors, method = c("BoxCox"))
trans

# Apply the transformation:
transformedPred<- predict(trans, filteredPredictors)
transformedPred

#checking skewness after transformation
skewValues <- apply(transformedPred, 2, skewness)
skewValues

#plotting histograms to see distribution
transformedPred %>%
  gather() %>% 
  ggplot(aes(x = value))+
  geom_histogram() +
  facet_wrap(~ key, scales = "free") +
  labs(x = NULL, y = NULL) +
  theme_bw() +
  theme(axis.ticks.y=element_blank())

#checking for outliers via box plots

transformedPred %>%
  gather() %>% 
  ggplot(aes(x ="", y = value))+
  geom_boxplot(outlier.colour = "lightblue", fill="yellow") +
  facet_wrap(~ key, scales = "free") +
  labs(x = NULL, y = NULL) +
  theme_bw() +
  theme(axis.ticks.y=element_blank())

#removing outliers by SpatialSign
#pre-processing transformations
trans <- preProcess(transformedPred, method = c("spatialSign"))
trans

# Apply the transformation:
transformedPred<- predict(trans, transformedPred)
transformedPred

#checking for outliers after transformation
transformedPred %>%
  gather() %>% 
  ggplot(aes(x ="", y = value))+
  geom_boxplot(outlier.colour = "lightblue", fill="yellow") +
  facet_wrap(~ key, scales = "free") +
  labs(x = NULL, y = NULL) +
  theme_bw() +
  theme(axis.ticks.y=element_blank())

###############################################
#checking distribution of classes to decide on splitting method
barplot(table(response$class))

set.seed(980)

#stratifed random sampling
trainingRows <- createDataPartition(response$class, p = .70, list= FALSE)
nrow(trainingRows)

#creating training and testing data

trainPredictors <- transformedPred [trainingRows, ]
trainClasses <- response[trainingRows]

str(trainClasses)

# Do the same for the test set using negative integers.
testPredictors <- transformedPred[-trainingRows, ]
testClasses <- response[-trainingRows]

str(trainPredictors)
str(testPredictors)
str(trainClasses)
str(testClasses)

nrow(trainPredictors)
nrow(testPredictors)

#summary of data

summary(trainPredictors)
sum(is.na(trainPredictors))

#checking frequency distribution of training and test classes
barplot(table(response$class))
barplot(table(trainClasses))
barplot(table(testClasses))

#model building

#1.Logistic Regression
set.seed(980)
ctrl <- trainControl(method = "cv", number = 10, summaryFunction = defaultSummary)
lrGrid <- expand.grid(.decay = seq(0, .4, length = 10))  ## use sequence till .8

set.seed(980)
lrFit <- train(x=trainPredictors,
               y = trainClasses,
               method = "multinom",
               metric = "Kappa",
               trControl = ctrl,
               tuneGrid = lrGrid)

lrFit
plot(lrFit)

predictiedLR<-predict(lrFit, testPredictors)
confusionMatrix(data = predictiedLR,
                reference = as.factor(testClasses))


#2.Linear Discriminant Analysis 
library(MASS)

set.seed(980)
ctrl <- trainControl(method = "cv", number = 10, summaryFunction = defaultSummary)

set.seed(980)
ldaFit <- train(x = trainPredictors,
                y = trainClasses,
                method = "lda",
                metric = "Kappa",
                preProc = c("center", "scale"),
                trControl = ctrl)
ldaFit

predictedLDA <- predict(ldaFit, testPredictors)
confusionMatrix(data = predictedLDA,
                reference = as.factor(testClasses))

#3.Partial Least Squares Discriminant 
set.seed(980)
ctrl <- trainControl(method = "cv", number = 10, summaryFunction = defaultSummary)

set.seed(980)
plsFit<- train(x = trainPredictors,
               y = trainClasses,
               method = "pls",
               tuneGrid = expand.grid(.ncomp = 1:8),
               preProc = c("center","scale"),
               metric = "Kappa",
               trControl = ctrl)
plsFit
plot(plsFit)

predictedPLS <-predict(plsFit, testPredictors)
confusionMatrix(data = predictedPLS,
                reference = as.factor(testClasses))

#4. Penalized Logistic Regression Model
plgGrid <- expand.grid(.alpha = c(.2, .4, .5, .6),
                      .lambda = seq(.0, .2, length = 5))   ##change tuning values to get a curve
set.seed(980)
ctrl <- trainControl(method = "cv", number = 10, summaryFunction = defaultSummary)

set.seed(980)
plgFit <- train(x=trainPredictors,
                y =trainClasses,
                method = "glmnet",
                tuneGrid = plgGrid,
                preProc = c("center", "scale"),
                metric = "Kappa",
                trControl = ctrl)

plgFit
plot(plgFit)

predictedPLG <-  predict(plgFit, testPredictors)
confusionMatrix(data = predictedPLG,
                reference = as.factor(testClasses))

#5.Sparse LDA 
library(sparseLDA)
ldaGrid <- expand.grid(.lambda = seq(.1, .2, length = 5), 
                       .NumVars = c(1:8))
set.seed(980)
ctrl <- trainControl(method = "cv", number = 10, summaryFunction = defaultSummary)

pldaFit <- train(x=trainPredictors,
                 y =trainClasses,
                 method = "sparseLDA",
                 importance = TRUE,
                 tuneGrid = ldaGrid,
                 preProc = c("center", "scale"),
                 metric = "Kappa",
                 trControl = ctrl)

pldaFit
plot(pldaFit)

predictedPLDA <-  predict(pldaFit, testPredictors)
confusionMatrix(data = predictedPLDA,
                reference = as.factor(testClasses))

#6. Nonlinear Discriminant Analysis
library(mda)
mdaGrid <- expand.grid(.subclasses = 1:10)

set.seed(980)
ctrl <- trainControl(method = "cv", number = 10, summaryFunction = defaultSummary)

set.seed(980)
mdaFit <- train(x=trainPredictors,
                 y =trainClasses,
                 method = "mda",
                 tuneGrid = mdaGrid,
                 metric = "Kappa",
                 trControl = ctrl)

mdaFit
plot(mdaFit)

predictedMDA <-  predict(mdaFit, testPredictors)
confusionMatrix(data = predictedMDA,
                reference = as.factor(testClasses))

#7.Neural Networks
library(nnet)

nnetGrid <- expand.grid(.size = 1:10, .decay = c(0, .1, 1, 2))
maxSize <- max(nnetGrid$.size)
numWts <- (maxSize * (8 + 1) + (maxSize+1)*2) ## 8 is the number of predictors

set.seed(980)
ctrl <- trainControl(method = "cv", number = 10, summaryFunction = defaultSummary)

set.seed(980)
nnetFit <- train(x=trainPredictors,
                 y =trainClasses,
                 method = "nnet",
                 metric = "Kappa",
                 preProc = c("center", "scale"),
                 tuneGrid = nnetGrid,
                 trace = FALSE,
                 maxit = 2000,
                 MaxNWts = numWts,
                 trControl = ctrl)
nnetFit
plot(nnetFit)

predictedNNET <-  predict(nnetFit, testPredictors)
confusionMatrix(data = predictedNNET,
                reference = as.factor(testClasses))

#8. Flexible Discriminant Analysis

fdaGrid <- expand.grid(.degree = 1:2, .nprune = 2:38)

set.seed(980)
ctrl <- trainControl(method = "cv", number = 10, summaryFunction = defaultSummary)

set.seed(980)
fdaFit <- train(x=trainPredictors,
                y =trainClasses,
                method = "fda",
                tuneGrid = fdaGrid,
                metric = "Kappa",
                trControl = ctrl)

fdaFit
plot(fdaFit)

predictedFDA <-  predict(fdaFit, testPredictors)
confusionMatrix(data = predictedFDA,
                reference = as.factor(testClasses))

#9. Support Vector Machines 

svmGrid<- expand.grid(.sigma = matrix(c(1, sigma, sigma, 2), 2, 2),
                               .C = 2^(seq(-4, 6)))

set.seed(980)
ctrl <- trainControl(method = "cv", number = 10, summaryFunction = defaultSummary)

set.seed(980)
svmFit <- train(x=trainPredictors,
                y =trainClasses,
                method = "svmRadial",
                tuneGrid = svmGrid,
                preProc = c("center", "scale"),
                metric = "Kappa",
                trControl = ctrl)

svmFit
plot(svmFit)

predictedSVM<-  predict(svmFit, testPredictors)
confusionMatrix(data = predictedSVM,
                reference = as.factor(testClasses))


#10. K-Nearest Neighbors ##
knnGrid<- data.frame(.k = 1:50)

set.seed(980)
ctrl <- trainControl(method = "cv", number = 10, summaryFunction = defaultSummary)

set.seed(980)
knnFit <- train(x=trainPredictors,
                y =trainClasses,
                method = "knn",
                metric = "Kappa",
                preProc = c("center", "scale"),
                tuneGrid = knnGrid,
                trControl = ctrl)
knnFit
plot(knnFit)

predictedKNN<-  predict(knnFit, testPredictors)
confusionMatrix(data = predictedKNN,
                reference = as.factor(testClasses))

#11 Naive Bayes 

nbGrid<- data.frame(.fL = 2,.usekernel = TRUE,.adjust = TRUE)

set.seed(980)
ctrl <- trainControl(method = "cv", number = 10, summaryFunction = defaultSummary)

set.seed(980)
nbFit <- train(x=trainPredictors,
                y =trainClasses,
                method = "nb",
                metric = "Kappa",
                preProc = c("center", "scale"),
                tuneGrid = nbGrid,
                trControl = ctrl)
nbFit
plot(nbFit)

predictedNB<-  predict(nbFit, testPredictors)
confusionMatrix(data = predictedNB,
                reference = as.factor(testClasses))



