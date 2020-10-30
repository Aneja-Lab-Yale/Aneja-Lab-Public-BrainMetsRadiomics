library(mRMRe)

survdata <- read.table("/Users/enochchang/brainmets/surv_labels.csv", header=TRUE, sep=",")
surv_object <- Surv(time = survdata$Time , event  = survdata$Event)
mydata <- read.table("/Users/enochchang/brainmets/UWA_radiomic.csv", header=TRUE, sep=",")[,-c(1)]
newdata <- cbind(surv_object, mydata)
dd <- mRMR.data(data = newdata)
filter <- mRMR.classic("mRMRe.Filter", data = dd, target_indices = c(1), feature_count = 40)
write.csv(solutions(filter), '/Users/enochchang/brainmets/surv_UWA_mRMRsolutions.csv', row.names = FALSE)

survdata <- read.table("/Users/enochchang/brainmets/surv_labels.csv", header=TRUE, sep=",")
surv_object <- Surv(time = survdata$Time , event  = survdata$Event)
mydata <- read.table("/Users/enochchang/brainmets/WA_radiomic.csv", header=TRUE, sep=",")[,-c(1)]
newdata <- cbind(surv_object, mydata)
dd <- mRMR.data(data = newdata)
filter <- mRMR.classic("mRMRe.Filter", data = dd, target_indices = c(1), feature_count = 40)
write.csv(solutions(filter), '/Users/enochchang/brainmets/surv_WA_mRMRsolutions.csv', row.names = FALSE)

survdata <- read.table("/Users/enochchang/brainmets/surv_labels.csv", header=TRUE, sep=",")
surv_object <- Surv(time = survdata$Time , event  = survdata$Event)
mydata <- read.table("/Users/enochchang/brainmets/big3_radiomic.csv", header=TRUE, sep=",")[,-c(1)]
newdata <- cbind(surv_object, mydata)
dd <- mRMR.data(data = newdata)
filter <- mRMR.classic("mRMRe.Filter", data = dd, target_indices = c(1), feature_count = 40)
write.csv(solutions(filter), '/Users/enochchang/brainmets/surv_big3_mRMRsolutions.csv', row.names = FALSE)

survdata <- read.table("/Users/enochchang/brainmets/surv_labels.csv", header=TRUE, sep=",")
surv_object <- Surv(time = survdata$Time , event  = survdata$Event)
mydata <- read.table("/Users/enochchang/brainmets/big1_radiomic.csv", header=TRUE, sep=",")[,-c(1)]
newdata <- cbind(surv_object, mydata)
dd <- mRMR.data(data = newdata)
filter <- mRMR.classic("mRMRe.Filter", data = dd, target_indices = c(1), feature_count = 40)
write.csv(solutions(filter), '/Users/enochchang/brainmets/surv_big1_mRMRsolutions.csv', row.names = FALSE)

survdata <- read.table("/Users/enochchang/brainmets/surv_labels.csv", header=TRUE, sep=",")
surv_object <- Surv(time = survdata$Time , event  = survdata$Event)
mydata <- read.table("/Users/enochchang/brainmets/smallest_radiomic.csv", header=TRUE, sep=",")[,-c(1)]
newdata <- cbind(surv_object, mydata)
dd <- mRMR.data(data = newdata)
filter <- mRMR.classic("mRMRe.Filter", data = dd, target_indices = c(1), feature_count = 40)
write.csv(solutions(filter), '/Users/enochchang/brainmets/surv_smallest_mRMRsolutions.csv', row.names = FALSE)

