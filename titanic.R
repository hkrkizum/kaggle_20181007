library(tidyverse)
library(mice)
library(caret)
library(xgboost)
library(doParallel)
detectCores()
registerDoParallel(makePSOCKcluster(8))

# Get rawdata --------------------------------------------------------------------
titanic_original <- read_csv("Rawdata/train.csv")
colnames(titanic_original)
titanic_original

# 学習用データの要約
summary(titanic_original)
apply(is.na(titanic_original), 2, sum)

# 欠損値の確認
md.pattern(titanic_original)

# PassengerID（乗客ID）,Name（名前）,Ticket（チケット番号）,Cabin（部屋番号）を除外
titanic_omit_varv <- titanic_original[, -c(1, 4, 9, 11)]
md.pattern(titanic_omit_varv)

# Embarkedの欠損値を除外
tmp <- titanic_omit_varv[!is.na(titanic_omit_varv$Embarked),]

# Ageをmiceで補完
tempData <- mice(tmp, method="pmm", m=5)
summary(tempData)

titanic_omit_NA <- mice::complete(tempData, 1)
md.pattern(titanic_omit_NA)

# Change data.frame to data.table 
rawdata <- data.table::data.table(titanic_omit_NA)
str(rawdata)
table(rawdata$Sex)
table(rawdata$Embarked)
rawdata$Survived <- as.factor(rawdata$Survived)

# str Sex to numetric: male 1, female 0 
rawdata[Sex=="male",Sex:="1"]
rawdata[Sex=="female",Sex:="2"]
rawdata$Sex <- as.numeric(rawdata$Sex)

# str Sex to numetric: male 1, female 0
rawdata[Embarked=="S",Embarked:="1"]
rawdata[Embarked=="Q",Embarked:="2"]
rawdata[Embarked=="S",Embarked:="3"]

# Make train data
n_rows <- nrow(rawdata)
index <- sample(n_rows, floor(0.8 * nrow(rawdata)))
rawdata_train <- rawdata[index,] #80%の学習データ
rawdata_test <- rawdata[-index,] #80%の学習データ


#trainControlを設定
ctrl <- trainControl(method = "cv",   
                     number = 10,
                     selectionFunction = "best")

#格子探索用のグリッド
grid <- expand.grid(nrounds = 500, 
                    eta = seq(0.01, 0.4, 0.01), 
                    max_depth = 4:10,
                    gamma = 0,
                    colsample_bytree = 1,
                    min_child_weight = 1,
                    subsample = 1)

#XGboostでパラメタチューニング
xgb.tune <- train(Survived ~ .,
                  data = rawdata_train,
                  method = "xgbTree",
                  metric = "Accuracy",
                  trControl = ctrl,
                  tuneGrid = grid)


plot(xgb.tune)  
xgb.tune$bestTune
result <- xgb.tune$results
round(result, 2) %>% 
    dplyr::filter(max_depth == 8)

xgb.pred <- predict(xgb.tune, rawdata_test)
confusionMatrix(data = xgb.pred,
                reference = rawdata_test$Survived,
                dnn = c("Prediction", "Actual"),
                mode = "prec_recall")

# TEST -----------------------------------------------------------------------------
titanic_test <- read_csv("Rawdata/test.csv") %>% as.data.frame()
dim(titanic_test)
colnames(titanic_test)
row.names(titanic_test) <- titanic_test$PassengerId

md.pattern(titanic_test)

titanic_test[is.na(titanic_test$Fare),]
titanic_test[is.na(titanic_test$Age),]

titanic_test[titanic_test$Ticket == 3701,]
titanic_test %>% 
    dplyr::select(-c(PassengerId, Cabin, Name)) -> tmp

# PassengerID（乗客ID）,Name（名前）,Ticket（チケット番号）,Cabin（部屋番号）を除外
titanic_test_omit_varv <- titanic_test[, -c(1, 3, 8, 10)]
md.pattern(titanic_test_omit_varv)

tempData <- mice(titanic_test_omit_varv, method="pmm", m=5)
summary(tempData)

titanic_test_omit_varv <- mice::complete(tempData, 1)
md.pattern(titanic_omit_NA)
dim(titanic_test_omit_varv)

# Change data.frame to data.table 
testdata <- data.table::data.table(titanic_test_omit_varv)
str(testdata)
table(testdata$Sex)
table(testdata$Embarked)

# str Sex to numetric: male 1, female 0 
testdata[Sex=="male",Sex:="1"]
testdata[Sex=="female",Sex:="2"]
testdata$Sex <- as.numeric(testdata$Sex)

# str Sex to numetric: male 1, female 0
testdata[Embarked=="S",Embarked:="1"]
testdata[Embarked=="Q",Embarked:="2"]
testdata[Embarked=="S",Embarked:="3"]

# Get predict data 
testdata_xgb.pred <- predict(xgb.tune, testdata)
testdata_xgb.pred

tmp <- data.frame(PassengerID = titanic_test$PassengerId, Survived = testdata_xgb.pred)
write.csv(tmp, file = 'Result/xgboostmodel.csv', row.names = F)
