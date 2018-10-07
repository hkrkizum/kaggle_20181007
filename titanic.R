library(tidyverse)
library(mice)
library(caret)
library(xgboost)

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
xgb.tune <- train(Old ~ .,
                  data = awabi_test,
                  method = "xgbTree",
                  metric = "Accuracy",
                  trControl = ctrl,
                  tuneGrid = grid)

xgb.tune$bestTune
nrounds max_depth  eta gamma colsample_bytree min_child_weight subsample
6     500         9 0.01     0                1                1         1
