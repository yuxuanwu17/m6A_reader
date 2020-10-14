# library(doParallel)
# registerDoParallel(makePSOCKcluster(5))

library(S4Vectors)
library(m6ALogisticModel)
library(SummarizedExperiment)
library(TxDb.Hsapiens.UCSC.hg19.knownGene)
library(BSgenome.Hsapiens.UCSC.hg19)
library(fitCons.UCSC.hg19)
library(phastCons100way.UCSC.hg19)
library(caret)
library(ROCR)
library(pROC)
library(MLmetrics)

set.seed(42)

source("~/m6A reader/featureSelection/F_score_fn.R")
source("/home/kunqi/m7G/method/class1.R")
source("/home/kunqi/m7G/method/class2.R")
source("/home/kunqi/m7G/method/class3.R")
source("/home/kunqi/m7G/method/class4.R")
source("/home/kunqi/m7G/method/class5.R")
source("/home/kunqi/m7G/method/class6.R")

FulleIF3a_Pos <- readRDS("/home/kunqi/m6A reader/eIF3a/Full/postive.rds")

# positive sample (all)
testT <- as.character(DNAStringSet(Views(Hsapiens, FulleIF3a_Pos + 20)))

# create the matrix to store the auc and roc value
aucMatrix <- matrix(NA, ncol = 1, nrow = 10)
prauc <- matrix(NA, ncol = 1, nrow = 10)
accMatrix <- matrix(NA, ncol = 1, nrow = 10)
mattMatrix <- matrix(NA, ncol = 1, nrow = 10)
storeNegMatrix <- matrix(NA, nrow = 10, ncol = 1)


# begin the loop
for (i in 1:2) {

  # get the negative sample
  storeNegMatrix[i,] <- paste("/home/kunqi/m6A reader/eIF3a/Full/negative", i, ".rds", sep = "")
  testN <- readRDS(storeNegMatrix[i,])

  #prepare the sample used in other encoding method as a character
  testN <- as.character(DNAStringSet(Views(Hsapiens, testN + 20)))

  # combine the two positive and negative sample together
  testAll <- c(testN, testT)

  # Make the predictoin
  source("/home/yuxuan.wu/m6A reader/encoding_method/onehot.R")

  # #convert it to the dataframe format
  dataframeR <- as.data.frame(matrixResults)

  # label the data
  label_train <- c(rep("Neg", (length(testN))), rep("Pos", (length(testT))))
  label_train <- factor(label_train, labels = c("Neg", "Pos"))

  #add the label to the dataframe
  dataframeR$class <- label_train

  #split the data
  intrain <- createDataPartition(y = dataframeR$class, p = 0.8, list = FALSE)
  training <- dataframeR[intrain,]
  testing <- dataframeR[-intrain,]

  fitControl <- trainControl(method = "cv",
                             number = 5,
                             classProbs = TRUE,
                             summaryFunction = twoClassSummary)

  reModel2 <- train(class ~ ., data = training[, c(top_selected_final, chemi_index, ncol(training))],
                    method = 'svmRadial',
                    preProc = c("center", "scale"),
                    trControl = fitControl,
                    metric = "ROC")

  try(testing$pred_class2 <- predict(reModel2, testing[, c(top_selected_final, chemi_index, (ncol(testing)))], type = "prob"))
  try(pred_result2 <- prediction(testing$pred_class2$Pos, testing$class))
  try(pred_result_auc2 <- performance(pred_result2, "auc"))
  try(aucMatrix[i,] <- attr(pred_result_auc2, "y.values")[[1]][1])
  testing$pred <- predict(reModel2, testing[, c(top_selected_final, chemi_index, (ncol(testing)))])

  # calculate the precision curve
  prauc[i,] <- PRAUC(testing$pred_class2$Pos, testing$class)

  # calculate the accurancy
  acc <- postResample(pred = testing$pred, obs = testing$class)[1]
  accMatrix[i,] <- acc

  # calculate the confusion matrix
  conf_matrix <- confusionMatrix(data = testing$pred, reference = testing$class, mode = "prec_recall")

  # calculate the matt_value
  source("/home/yuxuan.wu/m6A reader/encoding_method/Matt_value.R")
  matt <- Matt_Coef(conf_matrix)
  mattMatrix[i,] <- matt

}

avgAUC <- apply(aucMatrix, MARGIN = 2, FUN = mean, na.rm = TRUE)
indepTest <- avgAUC
indepTest

#caculate the max ROC of the prediction model
maxROC <- max(ROC_max)
crossValid <- maxROC
crossValid

# calculate the average of PRAUC
avgPRAUC <- apply(prauc, MARGIN = 2, FUN = mean, na.rm = TRUE)

# calculate the average of accuracy
avgACC <- apply(accMatrix, MARGIN = 2, FUN = mean, na.rm = TRUE)

# calculate the average of mtt value
avgMatt <- apply(mattMatrix, MARGIN = 2, FUN = mean, na.rm = TRUE)

results <- c(crossValid, indepTest, avgPRAUC, avgACC, avgMatt)


FulleIF3a_resultsMatx <- matrix(NA, nrow = 1, ncol = 5)
colnames(FulleIF3a_resultsMatx)<- c("crossValid", "indepTest", "avgPRAUC", "avgACC", "avgMatt")
FulleIF3a_resultsMatx[1,] <- results
saveRDS(FulleIF3a_resultsMatx, "/home/yuxuan.wu/m6A reader/5values/FulleIF3a_mixup.rds")


