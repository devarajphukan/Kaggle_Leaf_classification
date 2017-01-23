library("e1071")

# read data from csv
train_df <- read.csv("train.csv",header = TRUE)
test_df <- read.csv("test.csv",header = TRUE)

# column names
#names(train_df)

# Get Summary of all features
#summary(train_df)

test_ids <- test_df$id
train_species <- train_df$species

train_df <- within(train_df,rm("id"))
test_df <- within(test_df,rm("id"))
train_df <- within(train_df,rm("species"))

# Feature Scaling
scaleFunc <- function(x) {
  (x - mean(x))/sd(x)
}

train_df <- data.frame(lapply(train_df, scaleFunc))
test_df <- data.frame(lapply(test_df, scaleFunc))

# Information of classes
#print(levels(train_species))
#table(train_species) # ~ No class Imbalance

# Feature Importance
#set.seed(123)
#boruta.train <- Boruta::Boruta(train_df, train_species, doTrace = 2)
#print(boruta.train)

# Model Tuning

#svm_tune <- tune(svm, train.x=train_df, train.y=train_species,kernel="radial", ranges=list(cost=10^(-1:2),gamma=c(.5,1,2)))
#print(svm_tune)

# Model Prediction
svm_model <- svm(train_df,train_species,probability = TRUE)
pred <- predict(svm_model,test_df,probability = TRUE)

probs <- attr(pred,"probabilities")
result_df <- data.frame(probs)
result_df <- result_df[,order(names(result_df))]
result_df <- data.frame(id=test_ids,result_df)
write.csv(result_df,file="results1.csv",row.names = FALSE)








