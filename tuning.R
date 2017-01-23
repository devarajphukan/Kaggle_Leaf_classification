library("e1071")
library("mlbench")
library("caret")
library("doMC")
registerDoMC(cores = 4)

set.seed(123)
# read data from csv
train_df <- read.csv("train.csv",header = TRUE)
test_df <- read.csv("test.csv",header = TRUE)

# column names and class
#names(train_df)
#sapply(train_df,class)
# Get Summary of all features
#summary(train_df)

test_ids <- test_df$id
train_species <- train_df$species

train_df <- within(train_df,rm("id"))
test_df <- within(test_df,rm("id"))
train_df <- within(train_df,rm("species"))

# Information of classes
#print(levels(train_species))
#table(train_species) # ~ No class Imbalance

# Feature Scaling
scaleFunc <- function(x) {
	(x - mean(x))/sd(x)
}

train_df <- data.frame(lapply(train_df, scaleFunc))
test_df <- data.frame(lapply(test_df, scaleFunc))

# Removing Highly Correlated Features
getCorrelated <- function(x) {
	correlationMatrix <- cor(x)
	highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.75)
	return(highlyCorrelated)
}

trainCorrelated <- getCorrelated(train_df)
toRemove <- trainCorrelated[seq(1,length(trainCorrelated),2)]

train_df <- train_df[,-c(toRemove)]
test_df <- test_df[,-c(toRemove)]

# Top Features
control <- rfeControl(functions=rfFuncs, method="cv", number=2)
results <- rfe(train_df, train_species, rfeControl=control)
top_features <- results$optVariables#[1:50]
train_df_top <- train_df[top_features]
test_df_top <- test_df[top_features]

# Grid Search for the model
print("Tuning Model")
tune_rf <- tune(randomForest,
                train.x = train_df_top,
                train.y = train_species,
                ranges=list(ntree=c(500,1000,2000,5000),
                            mtry=c(.1,.2,.3),
                            nodesize=c(1,3,5,10),
                            localImp=c(TRUE,FALSE)))
print(tune_rf)
# rf <- randomForest(train_df_top, train_species, keep.forest=TRUE)
# probs <- predict(rf, test_df_top, type = "prob")
# 
# print(probs)
# probs <- data.frame(id=test_ids,probs)
# write.csv(probs,file="results2.csv",row.names = FALSE)
