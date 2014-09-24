train <- read.csv("~/Desktop/donorschoose/train.csv", na.strings="NaN")

train$projectid <- NULL
train <- train[c(sample(which(train$is_exciting == 0), sum(train$is_exciting == 1)), which(train$is_exciting == 1)),]

# require(biglm)
# lm1 <- bigglm(terms(is_exciting ~ ., data=train), data=train, family=binomial, chunksize=10, sandwich=TRUE)
lm1 <- glm(is_exciting ~ ., data=train, family=binomial)
# primary_focus_subject*resource_type + 
summary(lm1)

pred.train <- predict(lm1, train, type = "response")
require(ROCR)
performance(prediction(pred.train, train$is_exciting), 'auc')

test <- read.csv("~/Desktop/donorschoose/test.csv", na.strings="NaN")
test$teacher_prefix <- factor(replace(test$teacher_prefix, which(test$teacher_prefix=='Unknown'), c('Mrs.','Mrs.','Mrs.','Mrs.')))
pred.test <- predict(lm1, test, type = "response")
df <- data.frame(projectid=test$projectid, is_exciting=pred.test)
write.csv(df, file="~/Desktop/submission.csv", quote=FALSE, row.names=FALSE)