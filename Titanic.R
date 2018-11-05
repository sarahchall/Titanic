# Titanic survivors classification prediction 
# Sarah Hall
# Data sourced from Kaggle https://www.kaggle.com/c/titanic/data
# 3/11/2018

#load librarys
library(caret)
library(randomForest)
library (tidyverse)
library(fields)
library(s20x)

#load data files (downloaded from Kaggle)
TitanicTrain.df <- read.csv("train.csv", header = TRUE, stringsAsFactors = FALSE)
TitanicTest.df <- read.csv("test.csv", header = TRUE, stringsAsFactors = FALSE)

# view data sets
head(TitanicTrain.df)
head(TitanicTest.df)
#quick Data Quality Assessment
summary(TitanicTrain.df)
# Age contains NA's

#-------------------------------------------
# Exploratory Data Analysis and test for useful variables / features

names(TitanicTrain.df)
qt.df <- TitanicTrain.df %>%
  dplyr::select(Age,Fare)

pairs20x(qt.df)
# Age and Fare don't seem particualrly correlated

## Explore continuious variables
boxplot(TitanicTrain.df$Age ~ TitanicTrain.df$Survived)
# Age unlikely to be a useful predictor
boxplot(TitanicTrain.df$Fare ~ TitanicTrain.df$Survived)
# Fare could be useful

## Crosstabs for categorical variables
prop.table(table(TitanicTrain.df[,c("Survived", "Pclass")]),2)
#Pclass could be a useful predictor of Survived due to the survival ratio of each class
prop.table(table(TitanicTrain.df[,c("Survived", "Sex")]),2)
# Sex could be a useful predictor, higher % of women survived than men

prop.table(table(TitanicTrain.df[,c("Survived", "SibSp")]),2)
#SibSp could be useful, though unlikely

prop.table(table(TitanicTrain.df[,c("Survived", "Parch")]),2)
#Parch could be useful, though unlikely

prop.table(table(TitanicTrain.df[,c("Survived", "Embarked")]),2)
# Embarked looks useful


#-------------------------------------------
# Train model
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
# 79.9% accuracy looks good

## Try with Recursive Partitioning and Regressive Trees model
model.rpart <- train(Survived ~ Pclass + Sex + SibSp + 
                 Parch + Embarked + Fare,
               data = TitanicTrain.df,
               method = "rpart",
               trControl = trainControl(method = "cv", number = 10))
print (model.rpart)
# random forest is slightly better

#-------------------------------------------------
# Test model specific variable importance

varImp(model, useModel = TRUE, scale = TRUE)
plot(varImp(model, useModel = TRUE, scale = FALSE), top = 5)
# Sex is very important, Fare and Pclass are important and Parch and SibSp may be important (about the same impact)
# Test with Sex, Fare and Pclass, and another model with all but embarked

#-------------------------------------------------
# Train random forest model with Sex, Fare,Pclass, SibSp and Parch

model.FiveFt <- train(Survived ~ Pclass + Sex + SibSp + 
                 Parch + Fare,
               data = TitanicTrain.df,
               method = "rf",
               trControl = trainControl(method = "cv", number = 5))

print(model.FiveFt)
# Accuracy improved to 80%... very mild improvement


# Train random forest model with Sex, Fare and Pclass

model.ThreeFt <- train(Survived ~ Pclass + Sex + Fare,
                      data = TitanicTrain.df,
                      method = "rf",
                      trControl = trainControl(method = "cv", number = 5))

print(model.ThreeFt)
# Accuracy improved to 81%... slightly better improvement

#------------------------------
# Predict all four models on test set

summary(TitanicTest.df)
# NA value in "Fare"
# impute with mean of "Fare" col
TitanicTest.df$Fare <- ifelse(is.na(TitanicTest.df$Fare), mean(TitanicTest.df$Fare, na.rm = TRUE), TitanicTest.df$Fare)

# predict on test set
TitanicTest.df$Survived.RF_6FT <- predict(model, newdata = TitanicTest.df)
TitanicTest.df$Survived.RPRT_6FT <- predict(model.rpart, newdata = TitanicTest.df)
TitanicTest.df$Survived.RF_5FT <- predict(model.FiveFt, newdata = TitanicTest.df)
TitanicTest.df$Survived.RF_3FT <- predict(model.ThreeFt, newdata = TitanicTest.df)


# Create output files for Kaggle

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




#-------------------------------------------------------------------
## Next Steps to test:
## Is Random Forest the best model?
## Is there any interraction between Embarked and Pclass or Fare?
## Are Pclass and Fare proxys for each other?
## Are SibSp and Parch proxys for each other?

