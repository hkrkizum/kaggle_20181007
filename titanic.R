library(tidyverse)
library(mice)

# Get rawdata
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

rawdata[Embarked=="S",Embarked:="1"]
rawdata[Embarked=="S",Embarked:="1"]




