library(tidyverse)
library(mice)
library(caret)
library(xgboost)
library(randomForest)
library(stringr)
library(doParallel)
detectCores()
registerDoParallel(makePSOCKcluster(8))

# Get rawdata --------------------------------------------------------------------
titanic_original <- read_csv("Rawdata/train.csv") %>% data.table::data.table()
colnames(titanic_original)
titanic_original

# 学習用データの要約
summary(titanic_original)

# 欠損値の確認
md.pattern(titanic_original)

# PassengerID（乗客ID）,Ticket（チケット番号）除外
titanic_omit_varv <- titanic_original[, -c(1, 9)] %>% 
    dplyr::mutate(Salutation = 5, tmp ="", FamilySize = 0, IsAlone = 0) 
titanic_omit_varv <- data.table::data.table(titanic_omit_varv)

# familily name
# titanic_omit_varv$Fname <- str_extract(titanic_omit_varv$Name, ".*, ")
# titanic_omit_varv$Fname <- str_replace(titanic_omit_varv$Fname, ", ", "")
titanic_omit_varv$Fname <- sapply(titanic_omit_varv$Name, function(x) strsplit(x, split = '[,.]')[[1]][1])

# Salutation
# titanic_omit_varv$tmp <- str_replace(titanic_omit_varv$Name, ".*, ", "")
# titanic_omit_varv$tmp <- str_extract(titanic_omit_varv$tmp, ".*\\. ")
# titanic_omit_varv$tmp <- str_replace(titanic_omit_varv$tmp, "\\. ", "")
titanic_omit_varv$tmp <- gsub('(.*, )|(\\..*)', '', titanic_omit_varv$Name)

# Cahnge Salutation to num
titanic_omit_varv[tmp == "Mr", tmp:="1"]
titanic_omit_varv[tmp == "Miss", tmp:="2"]
titanic_omit_varv[tmp == "Mrs", tmp:="3"]
titanic_omit_varv[tmp == "Master", tmp:="4"]

titanic_omit_varv[tmp == "1", Salutation:=1]
titanic_omit_varv[tmp == "2", Salutation:=2]
titanic_omit_varv[tmp == "3", Salutation:=3]
titanic_omit_varv[tmp == "4", Salutation:=4]

unique(titanic_omit_varv$tmp)
unique(titanic_omit_varv$Salutation)

titanic_omit_varv$FamilySize <- titanic_omit_varv$SibSp + titanic_omit_varv$Parch + 1
titanic_omit_varv[FamilySize == 1, IsAlone:=1]

md.pattern(titanic_omit_varv)

colnames(titanic_omit_varv)
titanic_omit_varv_cut <- titanic_omit_varv %>% 
    dplyr::select(-c(3, 12))

# Embarkedの欠損値を代入
titanic_omit_varv_cut[is.na(titanic_omit_varv_cut$Embarked),]
titanic_omit_varv_cut[is.na(titanic_omit_varv_cut$Embarked), Embarked:="S"]

# Cabinの情報の有無で特徴量化
titanic_omit_varv_cut[!is.na(titanic_omit_varv_cut$Cabin), Cabin:="0"]
titanic_omit_varv_cut[is.na(titanic_omit_varv_cut$Cabin), Cabin:="1"]

# str Sex to numetric: male 1, female 0 
titanic_omit_varv_cut[Sex=="male",Sex:="0"]
titanic_omit_varv_cut[Sex=="female",Sex:="1"]

# str Sex to numetric: male 1, female 0
titanic_omit_varv_cut[Embarked=="S",Embarked:="1"]
titanic_omit_varv_cut[Embarked=="Q",Embarked:="2"]
titanic_omit_varv_cut[Embarked=="C",Embarked:="3"]

# Ageをmiceで補完
# tempData <- mice(titanic_omit_varv_cut, method="pmm", m=10)
# summary(tempData)
# rawdata <- mice::complete(tempData, 1)
# md.pattern(rawdata)

# Ageの欠損値に対する予測です。
predicted_age <- train(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Salutation + FamilySize,
                       # tuneGrid = data.frame(mtry = c(2, 3, 7)),
                       tuneLength = 4,
                       data = titanic_omit_varv_cut[!is.na(titanic_omit_varv_cut$Age), ],
                       method = "ranger",
                       trControl = trainControl(method = "cv", number = 10,repeats = 10, verboseIter = TRUE),importance = 'impurity')
plot(predicted_age)

predict(predicted_age, titanic_omit_varv_cut[is.na(titanic_omit_varv_cut$Age),])
titanic_omit_varv_cut[is.na(titanic_omit_varv_cut$Age), 
                      Age:=predict(predicted_age, 
                                   titanic_omit_varv_cut[is.na(titanic_omit_varv_cut$Age),])]
rawdata  <- titanic_omit_varv_cut

# Change data.frame to data.table 
rawdata$Survived <- as.factor(rawdata$Survived)
rawdata$Pclass <- as.factor(rawdata$Pclass)
rawdata$Sex <- as.factor(rawdata$Sex)
rawdata$Cabin <- as.factor(rawdata$Cabin)
rawdata$Embarked <- as.factor(rawdata$Embarked) 
rawdata$Salutation <- as.factor(rawdata$Salutation)
rawdata$IsAlone <- as.factor(rawdata$IsAlone)
rawdata$Fname <- as.factor(rawdata$Fname)

# rawdata$Survived <- as.factor(rawdata$Survived)
# rawdata$Pclass <- as.numeric(as.factor(rawdata$Pclass)) -1
# rawdata$Sex <- as.numeric(as.factor(rawdata$Sex)) -1
# rawdata$Cabin <- as.numeric(as.factor(rawdata$Cabin)) -1
# rawdata$Embarked <- as.numeric(as.factor(rawdata$Embarked)) -1 
# rawdata$Salutation <- as.numeric(as.factor(rawdata$Salutation)) -1
# rawdata$IsAlone <-as.numeric(as.factor(rawdata$IsAlone)) -1
# rawdata$Fname <- as.numeric(as.factor(rawdata$Fname)) - 1
str(rawdata)
md.pattern(titanic_omit_varv_cut)

# TEST -----------------------------------------------------------------------------
titanic_test <- read_csv("Rawdata/test.csv") %>% as.data.frame()
dim(titanic_test)
colnames(titanic_test)

md.pattern(titanic_test)

# PassengerID（乗客ID）,Ticket（チケット番号）除外
titanic_test_omit_varv <- titanic_test[, -c(1, 8)] %>% 
    dplyr::mutate(Salutation = 5, tmp ="", FamilySize = 0, IsAlone = 0) 
titanic_test_omit_varv <- data.table::data.table(titanic_test_omit_varv)

# familily name
titanic_test_omit_varv$Fname <- sapply(titanic_test_omit_varv$Name, function(x) strsplit(x, split = '[,.]')[[1]][1])


# Salutation
titanic_test_omit_varv$tmp <- gsub('(.*, )|(\\..*)', '', titanic_test_omit_varv$Name)

# Cahnge Salutation to num
titanic_test_omit_varv[tmp == "Mr", tmp:="1"]
titanic_test_omit_varv[tmp == "Miss", tmp:="2"]
titanic_test_omit_varv[tmp == "Mrs", tmp:="3"]
titanic_test_omit_varv[tmp == "Master", tmp:="4"]

titanic_test_omit_varv[tmp == "1", Salutation:=1]
titanic_test_omit_varv[tmp == "2", Salutation:=2]
titanic_test_omit_varv[tmp == "3", Salutation:=3]
titanic_test_omit_varv[tmp == "4", Salutation:=4]

unique(titanic_test_omit_varv$tmp)
unique(titanic_test_omit_varv$Salutation)

titanic_test_omit_varv$FamilySize <- titanic_test_omit_varv$SibSp + titanic_test_omit_varv$Parch + 1
titanic_test_omit_varv[FamilySize == 1, IsAlone:=1]

# Remove Name and tmp
colnames(titanic_test_omit_varv)
titanic_test_omit_varv_cut <- titanic_test_omit_varv %>% 
    dplyr::select(-c(2, 11))

md.pattern(titanic_test_omit_varv_cut)
# Fareに中央値
titanic_test_omit_varv_cut[is.na(titanic_test_omit_varv_cut$Fare),]
titanic_test_omit_varv_cut %>% 
    dplyr::filter(Pclass == 3 & Embarked == "S" & FamilySize == 1) %>% 
    dplyr::filter(!is.na(Fare)) %>% 
    summarise(medi = median(Fare))
titanic_test_omit_varv_cut[is.na(titanic_test_omit_varv_cut$Fare),Fare:=7.8958]

# str Sex to numetric: male 1, female 0 
titanic_test_omit_varv_cut[Sex=="male",Sex:="0"]
titanic_test_omit_varv_cut[Sex=="female",Sex:="1"]

# str Sex to numetric: male 1, female 0
titanic_test_omit_varv_cut[Embarked=="S",Embarked:="1"]
titanic_test_omit_varv_cut[Embarked=="Q",Embarked:="2"]
titanic_test_omit_varv_cut[Embarked=="C",Embarked:="3"]

# Cabinの情報の有無で特徴量化
titanic_test_omit_varv_cut[!is.na(titanic_test_omit_varv_cut$Cabin), Cabin:="1"]
titanic_test_omit_varv_cut[is.na(titanic_test_omit_varv_cut$Cabin), Cabin:="0"]

# Ageをmiceで補完
# tempData <- mice(titanic_test_omit_varv_cut, method="pmm", m=10)
# summary(tempData)
# testdata <- mice::complete(tempData, 1)
# md.pattern(testdata)
predicted_age <- train(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Salutation + FamilySize,
                       tuneLength = 4,
                       data = titanic_test_omit_varv_cut[!is.na(titanic_test_omit_varv_cut$Age), ],
                       method = "ranger",
                       trControl = trainControl(method = "cv", number = 10,repeats = 10, verboseIter = TRUE),importance = 'impurity')
plot(predicted_age)

predict(predicted_age, titanic_test_omit_varv_cut[is.na(titanic_test_omit_varv_cut$Age),])
titanic_test_omit_varv_cut[is.na(titanic_test_omit_varv_cut$Age), 
                      Age:=predict(predicted_age, 
                                   titanic_test_omit_varv_cut[is.na(titanic_test_omit_varv_cut$Age),])]

testdata <- titanic_test_omit_varv_cut

# Change Factor
testdata$Pclass <- as.factor(testdata$Pclass)
testdata$Sex <- as.factor(testdata$Sex)
testdata$Cabin <- as.factor(testdata$Cabin)
testdata$Embarked <- as.factor(testdata$Embarked)
testdata$Salutation <- as.factor(testdata$Salutation)
testdata$IsAlone <- as.factor(testdata$IsAlone)
testdata$Fname <- as.factor(testdata$Fname)

# testdata$Pclass <- as.numeric(as.factor(testdata$Pclass)) -1
# testdata$Sex <- as.numeric(as.factor(testdata$Sex)) -1
# testdata$Cabin <- as.numeric(as.factor(testdata$Cabin)) -1
# testdata$Embarked <- as.numeric(as.factor(testdata$Embarked)) -1 
# testdata$Salutation <- as.numeric(as.factor(testdata$Salutation)) -1
# testdata$IsAlone <-as.numeric(as.factor(testdata$IsAlone)) -1
# testdata$Fname <- as.numeric(as.factor(testdata$Fname)) - 1
str(testdata)

# Rundum forest ---------------------------------------------------------------------
# rf_model <- randomForest(factor(Survived) ~ .,data = rawdata)
# rf_model
# varImpPlot(rf_model)


# XGBoost ----------------------------------------------------------------------------
# # Make train data
n_rows <- nrow(rawdata)
index <- sample(n_rows, floor(0.8 * nrow(rawdata)))
rawdata_train <- rawdata[index,] #90%の学習データ
rawdata_test <- rawdata[-index,] 

str(rawdata)
rawdata_train %>% 
    dplyr::select(-Fname) -> rawdata_train
tmp1 <- dummyVars(~., data = rawdata_train)
tmp2 <- as.data.frame(predict(tmp1, rawdata_train))
tmp2 %>% 
    dplyr::select(-Survived.0,-Survived.1) -> tmp2
tmp2$Survived <- rawdata_train$Survived
rawdata_train <- tmp2

rawdata_test %>% 
    dplyr::select(-Fname) -> rawdata_test
tmp1 <- dummyVars(~., data = rawdata_test)
tmp2 <- as.data.frame(predict(tmp1, rawdata_test))
rawdata_test <- tmp2

# manual run
# y <- as.numeric(rawdata_train$Survived) -1
# x <- as.matrix(rawdata_train[, 2:12]) 
# set.seed(123)
# param = list("objective"="multi:softmax","num_class" = 2,"eval_metric" = "mlogloss")
# k=round(1+log2(nrow(x)))
# cv.nround = 100
# 
# bst.cv = xgb.cv(param=param, data = x, label = y, nfold = k,nrounds=cv.nround)
# 
# nround =8
# model = xgboost(param=param, data = x, label = y, nrounds=nround)
# 
# 
# test_x <- as.matrix(testdata) 
# test_x
# xgb.pred <- predict(model, test_x)
# xgb.pred

#trainControlを設定
ctrl <- trainControl(method = "cv",   
                     number = 10,
                     selectionFunction = "best",
                     verboseIter = TRUE)

#格子探索用のグリッド
grid <- expand.grid(nrounds = 500, 
                    eta = seq(0.01, 0.4, 0.01), 
                    max_depth = 4:10,
                    gamma = c(0,3,10),
                    colsample_bytree = 1,
                    min_child_weight = 1,
                    subsample = 1)
# 並列化実行
t<-proc.time()

cl <- makeCluster(detectCores())
registerDoParallel(cl)

#XGboostでパラメタチューニング
xgb.tune <- train(Survived ~ .,
                  data = rawdata_train,
                  method = "xgbTree",
                  metric = "Accuracy",
                  trControl = ctrl,
                  tuneGrid = grid,
                  verbose = TRUE)

stopCluster(cl)
proc.time()-t



plot(xgb.tune)  
xgb.tune$bestTune
result <- xgb.tune$results
round(result, 2) %>% 
    dplyr::filter(max_depth == 5 & eta == 0.36)

xgb.pred <- predict(xgb.tune, rawdata_test)
confusionMatrix(data = xgb.pred,
                reference = rawdata_test$Survived,
                dnn = c("Prediction", "Actual"),
                mode = "prec_recall")

str(rawdata_train)
rawdata_train
x <- as.matrix(rawdata_train[, -1])
bst <- xgboost(data = x,
               label = as.numeric(rawdata_train$Survived) -1,
               max.depth = 5,
               eta = 0.36,
               nthread = 8,
               nrounds = 10,
               objective = "multi:softmax",
               num_class = 2)
tmp <- predict(bst, as.matrix(rawdata_test[,-1]))

confusionMatrix(data = as.factor(tmp),
                reference = rawdata_test$Survived,
                dnn = c("Prediction", "Actual"),
                mode = "prec_recall")
xgb.importance(model = bst)
# Get predict data 
testdata_xgb.pred <- predict(xgb.tune, testdata)
testdata_xgb.pred

tmp <- data.frame(PassengerID = titanic_test$PassengerId, Survived = testdata_xgb.pred)
head(tmp)
write.csv(tmp, file = 'Result/xgboostmodel_4.csv', row.names = F)
