# Required Libraries ------------------------------------------------------

library(tm)
library(tidyverse)
library(RWeka)
library(doSNOW)
library(caret)
library(doParallel)
library(tidyverse)
library(LiblineaR)
library(ranger)
library(adabag)
library(xgboost)
library(earth)
library(mda)
library(logicFS)
library(rpart)


# Read in training data ---------------------------------------------------

data <- read.csv("Data/CleanMovie.csv", header = T)
View(data)
data$X <- NULL
df1 <- data[sample(nrow(data)),]
rownames(df1) <- NULL
table(df1$label)

df2 <- df1[1:10000, ]
nrow(df2)
corpus<- VCorpus(VectorSource(df2$text))
cleanDATA <- tm_map(corpus, removeWords, stopwords("english"))
cleanDATA <- tm_map(cleanDATA, content_transformer(stripWhitespace))


# Document Term Matrix  ------------------------------------------------------------------

#Unigram TF-IDF
dtm.tfidf <- DocumentTermMatrix(cleanDATA, control = list(weighting =function(x) weightTfIdf(x, normalize = FALSE)))
dtm.tfidf
dtm.tfidf.sparse <- removeSparseTerms(dtm.tfidf, 0.9994) # remove the sparsest terms 
dtm.tfidf.sparse

dtm <- DocumentTermMatrix(cleanDATA)
dtm
dtm.sparse <- removeSparseTerms(dtm.tfidf, 0.992)
dtm.sparse

remove(dtm, dtm.tfidf) # remove unneeded data to clear up memory fro training


# Data Partitioning ----------------------------------------------------------------

dataset <- dtm.tfidf.sparse

set.seed(1234)
trainIndex <- createDataPartition(df2$label, p = 0.8, list = FALSE, times = 1)

convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("Absent", "Present"))
}
dataset <- dataset %>% apply(MARGIN=2, FUN=convert_counts)

data_df_train <- dataset[trainIndex, ] %>% as.matrix() 
data_df_test <- dataset[-trainIndex, ] %>% as.matrix() 

data_df_train <- data.frame(data_df_train)
data_df_test <- data.frame(data_df_test)

train_label <- df2$label[trainIndex]
test_label <-df2$label[-trainIndex] 
train_label <- as.factor(train_label)
test_label <- as.factor(test_label)


# Fit Control Versions -------------------------------------------------------------

fitControl <- trainControl(method = "cv",
                           number = 5,
                           allowParallel = TRUE)

fitControl2 <- trainControl(method = "cv",
                           number = 1,
                           allowParallel = F)

fitControl3 <- trainControl(method = "cv",
                            number = 2,
                            allowParallel = T)


# Naive Bayes -------------------------------------------------------------
cl <- makeCluster(7)
registerDoParallel(cl)

model.nb <- train(x= data_df_train,
                  y= as.factor(train_label),
                  method = "nb",
                  trControl = fitControl)

pred.nb <- predict(model.nb,
                   newdata = data_df_test)

nb.cm <- confusionMatrix(pred.nb,as.factor(test_label), mode = "prec_recall")
nb.cm

stopCluster(cl)
registerDoSEQ()


# Stabalized Linear Discriminant Analysis -------------------------------------------------------------
cl <- makeCluster(7)
registerDoParallel(cl)

model.slda <- train(x= data_df_train,
                  y= as.factor(train_label),
                  method = "slda",
                  trControl = fitControl)

pred.slda <- predict(model.slda,
                   newdata = data_df_test)

slda.cm <- confusionMatrix(pred.slda, as.factor(test_label), mode = "prec_recall")
slda.cm

stopCluster(cl)
registerDoSEQ()


# Bagging -------------------------------------------------------------
cl <- makeCluster(7)
registerDoParallel(cl)

model.bag <- train(x= data_df_train,
                  y= as.factor(train_label),
                  method = "treebag",
                  trControl = fitControl3)

pred.bag <- predict(model.bag,
                   newdata = data_df_test)

bag.cm <- confusionMatrix(pred.bag, as.factor(test_label), mode = "prec_recall")
bag.cm

stopCluster(cl)
registerDoSEQ()


# CART -------------------------------------------------------------
cl <- makeCluster(7)
registerDoParallel(cl)

model.cart <- train(x= data_df_train,
                   y= as.factor(train_label),
                   method = "rpart",
                   trControl = fitControl)

pred.cart <- predict(model.cart,
                    newdata = data_df_test)

cart.cm <- confusionMatrix(pred.cart, as.factor(test_label), mode = "prec_recall")
cart.cm

stopCluster(cl)
registerDoSEQ()


# Support Vector Machine --------------------------------------------------
cl <- makeCluster(7)
registerDoParallel(cl)

model.svm <- train(x = data_df_train,
                 y = as.factor(train_label),
                 method = "svmLinearWeights2",
                 trControl = fitControl,
                 tuneGrid = data.frame(cost = 1, 
                                       Loss = 0, 
                                       weight = 1))

pred.svm <- predict(model.svm,
                    newdata = data_df_test)

svm.cm <- confusionMatrix(pred.svm, as.factor(test_label), mode = "prec_recall")
svm.cm

stopCluster(cl)
registerDoSEQ()


# Logit Boost -------------------------------------------------------------


model.lb <- train(x = data_df_train,
                        y = as.factor(train_label),
                        method = "LogitBoost",
                        trControl = fitControl)

pred.lb <- predict(model.lb,
                           newdata = data_df_test)

logitboost.cm <- confusionMatrix(pred.lb, as.factor(test_label), mode = "prec_recall")
logitboost.cm


# Random Forests ----------------------------------------------------------
cl <- makeCluster(7)
registerDoParallel(cl)

model.rf <- train(x = data_df_train, 
                y = as.factor(train_label), 
                method = "ranger",
                trControl = fitControl,
                tuneGrid = data.frame(mtry = floor(sqrt(dim(data_df_train)[2])),
                                      splitrule = "gini",
                                      min.node.size = 1))


pred.rf <- predict(model.rf,
                   newdata = data_df_test)

rf.cm <- confusionMatrix(pred.rf, as.factor(test_label), mode = "prec_recall")
rf.cm


stopCluster(cl)
registerDoSEQ()


# NN ----------------------------------------------------------------------
cl <- makeCluster(7)
registerDoParallel(cl)

model.nnet <- train(x = data_df_train,
                  y = as.factor(train_label),
                  method = "nnet",
                  trControl = fitControl,
                  tuneGrid = data.frame(size = 4,
                                        decay = 5e-4),
                  MaxNWts = 5000)

pred.nnet <- predict(model.nnet,
                     newdata = data_df_test)

nnet.cm <- confusionMatrix(pred.nnet, as.factor(test_label), mode = "prec_recall")
nnet.cm

stopCluster(cl)
registerDoSEQ()


# Model Comparison  -------------------------------------------------------

mod_results <- rbind(
  svm.cm$overall, 
  nb.cm$overall,
  logitboost.cm$overall,
  rf.cm$overall,
  nnet.cm$overall,
  bag.cm$overall,
  cart.cm$overall, 
  slda.cm$overall
) %>%
  as.data.frame() %>%
  mutate(model = c("SVM", "Naive-Bayes", "LogitBoost", "Random forest", "Neural network", "Bagging", "CART", "SLDA"))


# Comparison Visualization ------------------------------------------------

mod_results %>%
  ggplot(aes(model, Accuracy)) +
  geom_point() +
  ylim(0, 1) +
  geom_hline(yintercept = mod_results$AccuracyNull[1],
             color = "red")+
  theme(text = element_text(size=15),
        axis.text.x = element_text(angle=10, hjust = 1))+
  labs(title="Model Performance Overview Movie Reviews", x="Models")

##Save high resolution of plot
tiff (filename='ModelPerformanceTweets.tiff',  width=4, height=6 ,units="in",res=300)
mod_results %>%
  ggplot(aes(model, Accuracy)) +
  geom_point() +
  ylim(0, 1) +
  geom_hline(yintercept = mod_results$AccuracyNull[1],
             color = "red")+
  theme(text = element_text(size=15),
        axis.text.x = element_text(angle=90, hjust = 1))+
  labs(title="Performance Tweets", x="Models")
dev.off ()


# Prediction with Random Forests --------------------------------------------------------------

testtweets <- read.csv("CleanTweets.csv", fileEncoding = "UTF-8")
View(testtweets)
testtweets$X <- NULL
testtweets$text <- as.character(testtweets$text)

corpus<- VCorpus(VectorSource(testtweets$text))
cleanDATA <- tm_map(corpus, removeWords, stopwords("english"))
cleanDATA <- tm_map(cleanDATA, content_transformer(stripWhitespace))


#Unigram TF-IDF
dtm.tfidf <- DocumentTermMatrix(cleanDATA, control = list(weighting =function(x) weightTfIdf(x, normalize = FALSE)))
dtm.tfidf
dtm.tfidf.sparse <- removeSparseTerms(dtm.tfidf, 0.99952)
dtm.tfidf.sparse

remove(dtm, dtm.tfidf)

dataset <- dtm.tfidf.sparse
convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("Absent", "Present"))
}
dataset <- dataset %>% apply(MARGIN=2, FUN=convert_counts)

testdata <- as.matrix(dataset)
testdata <- data.frame(testdata)


names(testdata) <- make.names(names(testdata))


rf.pred <- predict(model.rf, newdata =testdata[, 1:1430] )

resultlist <- data.frame(rf.pred)
testtweets$label <- resultlist
