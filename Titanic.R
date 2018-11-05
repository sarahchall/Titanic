#' ---
#' title: Titanic survivors classification prediction 
#' author: Sarah Hall
#' subtitle: Data sourced from Kaggle https://www.kaggle.com/c/titanic/data
#' date: 3/11/2018
#' ---

#load librarys
library(caret)
library(randomForest)
library (tidyverse)
library(fields)
library(s20x)

#load data files (downloaded from Kaggle)
TitanicTrain.df <- read.csv("train.csv", header = TRUE, stringsAsFactors = FALSE)
TitanicTest.df <- read.csv("test.csv", header = TRUE, stringsAsFactors = FALSE)

#' View data sets:
head(TitanicTrain.df)
head(TitanicTest.df)
#' Quick data quality assessment:
#' ------------------------------
summary(TitanicTrain.df)
#' Age contains NA's, will need to impute these if using age as a feature.
#'
#' Exploratory data analysis
#' -----------------------------------------------------------------
#' Test for useful features:

names(TitanicTrain.df)
qt.df <- TitanicTrain.df %>%
  dplyr::select(Age,Fare)

pairs20x(qt.df)
#' Age and Fare don't seem particualrly correlated.
#'
#' Explore continuious variables:
boxplot(TitanicTrain.df$Age ~ TitanicTrain.df$Survived)
#' Age unlikely to be a useful predictor, will not use as feature.
boxplot(TitanicTrain.df$Fare ~ TitanicTrain.df$Survived)
#' Fare could be useful, will use as feature.
#'
#' Explore categorical variables:
prop.table(table(TitanicTrain.df[,c("Survived", "Pclass")]),2)
#' Pclass could be a useful predictor due to the survival ratio of each class, will use as a feature.
prop.table(table(TitanicTrain.df[,c("Survived", "Sex")]),2)
#' Sex could be a useful predictor, higher % of women survived than men, will use as a feature.
prop.table(table(TitanicTrain.df[,c("Survived", "SibSp")]),2)
#' SibSp could be useful, though unlikely. Will use as a feature initially.
prop.table(table(TitanicTrain.df[,c("Survived", "Parch")]),2)
#' Parch could be useful, though unlikely. Will use as a feature initially.
prop.table(table(TitanicTrain.df[,c("Survived", "Embarked")]),2)
#' Embarked looks useful, will use as a feature.
#' Name, Cabin, and Ticket will not be used as features.


#' 
#' Initial model train: random forest with six features
#' ----------------------------------------------------

# Convert Survived to factor
TitanicTrain.df$Survived = factor(TitanicTrain.df$Survived)
# Set a random seed
set.seed(123)
# Train the model - random forest
model <- train(Survived ~ Pclass + Sex + SibSp + 
                Parch + Embarked + Fare,
               data = TitanicTrain.df,
               method = "rf",
               trControl = trainControl(method = "cv", number = 5))
print(model)
#' 79% accuracy looks good for a first stab.

#'
#' Test model specific variable importance
#' -----------------------------------------

varImp(model, useModel = TRUE, scale = TRUE)
plot(varImp(model, useModel = TRUE, scale = FALSE), top = 5)
#' Sex is a very important feature, Fare and Pclass are important and Parch and SibSp may be important (about the same impact as each other).
#' 
#' Test two new models, both random forest. One with Sex, Fare and Pclass as features, the other with SibSp and Parch as well.
#' 
#' Train random forest model with Sex, Fare,Pclass, SibSp and Parch
#' ---------------------------------------------------------------
model.FiveFt <- train(Survived ~ Pclass + Sex + SibSp + 
                 Parch + Fare,
               data = TitanicTrain.df,
               method = "rf",
               trControl = trainControl(method = "cv", number = 5))
print(model.FiveFt)
#' Accuracy improved to 82%, good improvement.

#'
#' Train random forest model with Sex, Fare and Pclass
#' ---------------------------------------------------
model.ThreeFt <- train(Survived ~ Pclass + Sex + Fare,
                      data = TitanicTrain.df,
                      method = "rf",
                      trControl = trainControl(method = "cv", number = 5))
print(model.ThreeFt)
#' Accuracy of 81% is a slight improvement on initial 6 feature model, but the 5 feature model is best so far.
#'
#' Test with different model: recursive partitioning and regressive trees model
#' ----------------------------------------------------------------------------
model.rpart <- train(Survived ~ Pclass + Sex + SibSp + 
                       Parch + Embarked + Fare,
                     data = TitanicTrain.df,
                     method = "rpart",
                     trControl = trainControl(method = "cv", number = 5))
print (model.rpart)
#' Accuracy of 81% about on par with three feature random forest model.
#' 
#' Five feature random forest is still best.

#'
#' Predict all four models on the test set of data
#' -----------------------------------------------

summary(TitanicTest.df)
#' NA value in "Fare" column.
#' Fix this by imputing with mean of "Fare" column.
TitanicTest.df$Fare <- ifelse(is.na(TitanicTest.df$Fare), mean(TitanicTest.df$Fare, na.rm = TRUE), TitanicTest.df$Fare)

# predict on test set
TitanicTest.df$Survived.RF_6FT <- predict(model, newdata = TitanicTest.df)
TitanicTest.df$Survived.RPRT_6FT <- predict(model.rpart, newdata = TitanicTest.df)
TitanicTest.df$Survived.RF_5FT <- predict(model.FiveFt, newdata = TitanicTest.df)
TitanicTest.df$Survived.RF_3FT <- predict(model.ThreeFt, newdata = TitanicTest.df)


#' Create output files for Kaggle
#' ------------------------------

output_RF_6FT <- TitanicTest.df %>%
  dplyr::select(PassengerId,Survived.RF_6FT)
colnames(output_RF_6FT)[2] <- "Survived"
write.csv(output_RF_6FT, file = "output_RF_6FT.csv", row.names = FALSE)

output_RPRT_6FT <- TitanicTest.df %>%
  dplyr::select(PassengerId,Survived.RPRT_6FT)
colnames(output_RPRT_6FT)[2] <- "Survived"
write.csv(output_RPRT_6FT, file = "output_RPRT_6FT.csv", row.names = FALSE)

output_RF_5FT <- TitanicTest.df %>%
  dplyr::select(PassengerId,Survived.RF_5FT)
colnames(output_RF_5FT)[2] <- "Survived"
write.csv(output_RF_5FT, file = "output_RF_5FT.csv", row.names = FALSE)

output_RF_3FT <- TitanicTest.df %>%
  dplyr::select(PassengerId,Survived.RF_3FT)
colnames(output_RF_3FT)[2] <- "Survived"
write.csv(output_RF_3FT, file = "output_RF_3FT.csv", row.names = FALSE)


#' Summary
#' -------
#' 
#' **Results from Kaggle:**
#' 
#' * Random forest with 6 features: 77.9% accuracy
#' * Random forest with 5 features: 79.4% accuracy
#' * Random forest with 3 features: 77.5% accuracy
#' * Recursive partitioning and regressive trees model with 6 features: 78.4% accuracy
#' 
#' As expected, the random forest model with 5 features (Sex, Pclass, Fare, SibSp and Parch) performed best.
#' 
#' No model provided a significant imporvement on the initial model though, so next steps will be required to improve my score.
#' 
#' **Next steps to test:**
#' 
#' * Is random forest the best model?
#' * Is there any interraction between Embarked and Pclass or Fare?
#' * Can overfitting be reduced?
#'     + Are Pclass and Fare correlated?
#'     + Are SibSp and Parch correlated?

