#### Exam 2024 ####

setwd("C:/Users/andre/Documents/3. Semester Kandidaten/Decision Making/Eksamen")

getwd()


####  Part 1 ####
library(dplyr)
library(fastDummies)

#### Data Processing ####
student <- read.table("student.txt", header=TRUE, stringsAsFactors = TRUE)
studentadditional <- read.table("studentadditional.txt", header=TRUE, stringsAsFactors = TRUE)

student$class <- as.factor(student$class)
studentadditional$class <- as.factor(studentadditional$class)

dummy_cols_student <- dummy_cols(student, select_columns = c("sex", "address", "famsize", "Pstatus",
                                                             "schoolsup", "famsup", "paid", "activities",
                                                             "nursery", "higher", "romantic"),
                                 remove_selected_columns = TRUE )

dummy_cols_studentadditional <- dummy_cols(studentadditional, select_columns = c("sex", "address", "famsize",
                                                                                 "Pstatus", "schoolsup",
                                                                                 "famsup", "paid", "activities",
                                                                                 "nursery", "higher", "romantic"),
                                           remove_selected_columns = TRUE)

studentND <- dummy_cols_student
studentadditionalND <- dummy_cols_studentadditional


for (i in 1:(ncol(dummy_cols_student) - 1)) {
  if (class(dummy_cols_student[, i]) == "numeric" || class(dummy_cols_student[, i]) == "integer") {
    minimum <- min(dummy_cols_student[, i])
    maximum <- max(dummy_cols_student[, i])
    studentND[, i] <- as.vector(scale(dummy_cols_student[, i], center = minimum, scale = maximum - minimum))
    studentadditionalND[, i] <- as.vector(scale(dummy_cols_studentadditional[, i], center = minimum, scale = maximum - minimum))
  }
}

studentND$romantic_yes <- as.numeric(studentND$romantic_yes)

studentND$class
class_counts <- table(studentND$class)

bar_positions <- barplot(class_counts, 
                         main = "Frequency of High and Low Categories", 
                         col = c("lightblue", "lightcoral"),
                         ylab = "Frequency",
                         xlab = "Category",
                         ylim = c(0, max(class_counts) + 10)) 

text(x = bar_positions, 
     y = class_counts, 
     labels = class_counts, 
     pos = 3, 
     cex = 0.8, 
     col = "black")

str(studentND)
str(studentadditionalND)

#### Part 1 i ####

set.seed(2024)
full_model <- glm(class ~ ., data = studentND, family = binomial)
stepwise_model <- step(full_model)

train_predictions <- predict(stepwise_model, newdata = studentND, type = "response")
train_pred_class <- ifelse(train_predictions > 0.5, "Low", "High")

train_conf_matrix <- table(studentND$class, train_pred_class)
train_accuracy <- mean(studentND$class == train_pred_class)

coefficients <- coef(stepwise_model)
print(coefficients)

print("Confusion Matrix - Training Data (student):")
print(train_conf_matrix)
print(paste("Accuracy - Training Data:", round(train_accuracy * 100, 2), "%"))



test_predictions <- predict(stepwise_model, newdata = studentadditionalND, type = "response")
test_pred_class <- ifelse(test_predictions > 0.5, "Low", "High")

test_conf_matrix <- table(studentadditionalND$class, test_pred_class)
test_accuracy <- mean(studentadditionalND$class == test_pred_class)

print("Confusion Matrix - Training Data (studentadditional):")
print(test_conf_matrix)
print(paste("Accuracy - Training Data:", round(test_accuracy * 100, 2), "%"))

train_accuracy_logistic <-train_accuracy
train_accuracy_logistic
test_accuracy_logistic <- test_accuracy
test_accuracy_logistic

# Appendix 
# Odds Ratio Plot
library(ggplot2)
coeff <- exp(coef(stepwise_model))
conf <- exp(confint(stepwise_model))
odds_df <- data.frame(
  Variable = names(coeff),
  OR = coeff,
  LowerCI = conf[,1],
  UpperCI = conf[,2]
)
ggplot(odds_df, aes(x = reorder(Variable, OR), y = OR)) +
  geom_point() +
  geom_errorbar(aes(ymin = LowerCI, ymax = UpperCI), width = 0.2) +
  coord_flip() +
  labs(title = "Odds Ratios with 95% CI", y = "Odds Ratio", x = "Variables")

# ROC Curve tjek lige. ser lidt underligt ud med en højere Area under the curve end præcicision.
library(pROC)
roc_obj <- roc(studentND$class, train_predictions)
plot(roc_obj, main="ROC Curve for Logistic Regression")
auc(roc_obj)



#### Part 1 ii ####
library(e1071)
library(ggplot2)
library(corrplot)

# SVM Analysis
set.seed(2024)

# ---- LINEAR KERNEL ----

cat("Testing kernel: linear\n")
tuned_linear <- tune.svm(class ~ ., data = studentND, kernel = "linear", cost = 2^(-5:5))
best_model_linear <- tuned_linear$best.model
best_cost_linear <- tuned_linear$best.parameters$cost
train_accuracy_linear <- sum(diag(table(predict(best_model_linear, studentND), studentND$class))) / nrow(studentND)
test_accuracy_linear <- sum(diag(table(predict(best_model_linear, studentadditionalND), studentadditionalND$class))) / nrow(studentadditionalND)
cat("Best cost for linear kernel:", best_cost_linear, "\n")
cat("Training accuracy for linear kernel:", round(train_accuracy_linear * 100, 2), "%\n")
cat("Test accuracy for linear kernel:", round(test_accuracy_linear * 100, 2), "%\n\n")

train_accuracy_svm_linear <- train_accuracy_linear
test_accuracy__svm_linear <- test_accuracy_linear

summary(best_model_linear)


set.seed(2024)
# ---- POLYNOMIAL KERNEL ----
cat("Testing kernel: polynomial\n")
tuned_polynomial <- tune.svm(class ~ ., data = studentND, kernel = "polynomial", 
                             cost = 2^(-12:12), degree = 2:4,
                             coef0 =0:20)
best_model_polynomial <- tuned_polynomial$best.model
best_cost_polynomial <- tuned_polynomial$best.parameters$cost
best_degree_polynomial <- tuned_polynomial$best.parameters$degree
best_coef0_polynomial <- tuned_polynomial$best.parameters$coef0
train_accuracy_polynomial <- sum(diag(table(predict(best_model_polynomial, studentND), studentND$class))) / nrow(studentND)
test_accuracy_polynomial <- sum(diag(table(predict(best_model_polynomial, studentadditionalND), studentadditionalND$class))) / nrow(studentadditionalND)
cat("Best cost for polynomial kernel:", best_cost_polynomial, "\n")
cat("Best degree for polynomial kernel:", best_degree_polynomial, "\n")
cat("Best coef0 for polynomial kernel:", best_coef0_polynomial, "\n")
cat("Training accuracy for polynomial kernel:", round(train_accuracy_polynomial * 100, 2), "%\n")
cat("Test accuracy for polynomial kernel:", round(test_accuracy_polynomial * 100, 2), "%\n\n")

best_model_polynomial$gamma

train_accuracy_svm_polynomial <- train_accuracy_polynomial
test_accuracy__svm_polynomial <- test_accuracy_polynomial

# Confusion matrix for training data
cat("Confusion Matrix for Training Data:\n")
train_predictions <- predict(best_model_polynomial, studentND)
train_confusion_matrix <- table(Predicted = train_predictions, Actual = studentND$class)
print(train_confusion_matrix)

# Confusion matrix for test data
cat("\nConfusion Matrix for Test Data:\n")
test_predictions <- predict(best_model_polynomial, studentadditionalND)
test_confusion_matrix <- table(Predicted = test_predictions, Actual = studentadditionalND$class)
print(test_confusion_matrix)

set.seed(2024)
# ---- RADIAL KERNEL ----
cat("Testing kernel: radial\n")
tuned_radial <- tune.svm(class ~ ., data = studentND, kernel = "radial", cost = 2^(-7:7), gamma = 2^(-7:7))
best_model_radial <- tuned_radial$best.model
best_cost_radial <- tuned_radial$best.parameters$cost
best_gamma_radial <- tuned_radial$best.parameters$gamma
train_accuracy_radial <- sum(diag(table(predict(best_model_radial, studentND), studentND$class))) / nrow(studentND)
test_accuracy_radial <- sum(diag(table(predict(best_model_radial, studentadditionalND), studentadditionalND$class))) / nrow(studentadditionalND)
cat("Best cost for radial kernel:", best_cost_radial, "\n")
cat("Best gamma for radial kernel:", best_gamma_radial, "\n")
cat("Training accuracy for radial kernel:", round(train_accuracy_radial * 100, 2), "%\n")
cat("Test accuracy for radial kernel:", round(test_accuracy_radial * 100, 2), "%\n\n")

train_accuracy_svm_radial <- train_accuracy_radial
test_accuracy__svm_radial <- test_accuracy_radial

set.seed(2024)
# ---- SIGMOID KERNEL ----
cat("Testing kernel: sigmoid\n")
tuned_sigmoid <- tune.svm(class ~ ., data = studentND, kernel = "sigmoid", cost = 2^(-7:7),
                          gamma = 2^(-7:7), coef0 = 0:10)
best_model_sigmoid <- tuned_sigmoid$best.model
best_cost_sigmoid <- tuned_sigmoid$best.parameters$cost
best_gamma_sigmoid <- tuned_sigmoid$best.parameters$gamma
best_coef0_sigmoid <- tuned_sigmoid$best.parameters$coef0
train_accuracy_sigmoid <- sum(diag(table(predict(best_model_sigmoid, studentND), studentND$class))) / nrow(studentND)
test_accuracy_sigmoid <- sum(diag(table(predict(best_model_sigmoid, studentadditionalND), studentadditionalND$class))) / nrow(studentadditionalND)
cat("Best cost for sigmoid kernel:", best_cost_sigmoid, "\n")
cat("Best gamma for sigmoid kernel:", best_gamma_sigmoid, "\n")
cat("Best coef0 for sigmoid kernel:", best_coef0_sigmoid, "\n")
cat("Training accuracy for sigmoid kernel:", round(train_accuracy_sigmoid * 100, 2), "%\n")
cat("Test accuracy for sigmoid kernel:", round(test_accuracy_sigmoid * 100, 2), "%\n\n")

train_accuracy_svm_sigmoid <- train_accuracy_sigmoid
test_accuracy__svm_sigmoid <- test_accuracy_sigmoid

# Visualization

# Distribution of class labels
ggplot(studentND, aes(x = class)) +
  geom_bar(fill = "skyblue") +
  labs(title = "Distribution of Class Labels", x = "Class", y = "Count") +
  theme_minimal()

# Correlation matrix for continuous variables
continuous_vars <- c("age", "Medu", "Fedu", "traveltime", "studytime", "failures", 
                     "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences")
student_continuous <- studentND[, continuous_vars]
cor_matrix <- cor(student_continuous)
corrplot(cor_matrix, method = "color", type = "upper", tl.col = "black", tl.srt = 45)


# Part 1 iii 
set.seed(2024)

library(tree)
library(rpart)
library(rpart.plot)

mytree <- tree(class ~ ., studentND)
plot(mytree)
text(mytree)
summary(mytree)

mypredictiontree_training <- predict(mytree, studentND, type = "class")
classificationtable_training <- table(mypredictiontree_training, studentND$class)
print("Confusion Matrix - Traning data (Original tree):")
print(classificationtable_training)
mytree_training_pred <- sum(diag(classificationtable_training)) / sum(classificationtable_training)
cat("Traning Accuracy (Original tree):", round(mytree_training_pred * 100, 2), "%\n\n")

mypredictiontree_test <- predict(mytree, studentadditionalND, type = "class")
classificationtable_test <- table(mypredictiontree_test, studentadditionalND$class)
print("Confusion Matrix - Test Data (Original Pruning):")
print(classificationtable_test)
acctesttree_test <- sum(diag(classificationtable_test)) / sum(classificationtable_test)
cat("Test Accuracy (Original tree):", round(acctesttree_test * 100, 2), "%\n\n")

train_accuracy_tree_standard <- mytree_training_pred
test_accuracy__tree_standard <- acctesttree_test

set.seed(2024)
mycrossval <- cv.tree(mytree, FUN = prune.tree)
mybestsize <- mycrossval$size[which(mycrossval$dev == min(mycrossval$dev))]
myprunedtree_old <- prune.tree(mytree, best = mybestsize[1])

plot(myprunedtree_old)
text(myprunedtree_old)
title("Pruned Tree Using prune.tree")

myprunedtree_old_training <- predict(myprunedtree_old, studentND, type = "class")
classificationtable_training <- table(myprunedtree_old_training, studentND$class)
print("Confusion Matrix - Test Data (Original Pruning):")
print(classificationtable_training)

acctesttree_training <- sum(diag(classificationtable_training)) / sum(classificationtable_training)
acctesttree_training

mypredictiontree_test_old <- predict(myprunedtree_old, studentadditionalND, type = "class")
classificationtable_test_old <- table(mypredictiontree_test_old, studentadditionalND$class)
print("Confusion Matrix - Test Data (Original Pruning):")
print(classificationtable_test_old)
acctesttree_test_old <- sum(diag(classificationtable_test_old)) / sum(classificationtable_test_old)
cat("Test Accuracy (Original Pruning):", round(acctesttree_test_old * 100, 2), "%\n\n")

train_accuracy_tree_sizedpruned <- acctesttree_training
test_accuracy__tree_sizedpruned <- acctesttree_test_old

mytree_rpart <- rpart(class ~ ., data = studentND, method = "class", control = rpart.control(cp = 0.0))

rpart.plot(mytree_rpart)
title("Original Tree with rpart")

# Cost-complexity pruning
printcp(mytree_rpart) 
optimal_cp <- mytree_rpart$cptable[which.min(mytree_rpart$cptable[, "xerror"]), "CP"]
myprunedtree_new_cost <- prune(mytree_rpart, cp = optimal_cp)
# Plot tuned pruning tree
rpart.plot(myprunedtree_new)
title("Pruned Tree Using Cost-Complexity Pruning")

plot(myprunedtree_new_cost)
text(myprunedtree_new_cost)
title("Pruned Tree Using Cost-Complexity Pruning")

mypredictiontree_training_new <- predict(myprunedtree_new_cost, studentND, type = "class")
classificationtable_training_new <- table(mypredictiontree_training_new, studentND$class)
print("Confusion Matrix - Training Data (Cost-Complexity Pruning):")
print(classificationtable_training_new)
acctesttree_training_new <- sum(diag(classificationtable_training_new)) / sum(classificationtable_training_new)
cat("Training Accuracy (Cost-Complexity Pruning):", round(acctesttree_training_new * 100, 2), "%\n")

mypredictiontree_test_new <- predict(myprunedtree_new_cost, studentadditionalND, type = "class")
classificationtable_test_new <- table(mypredictiontree_test_new, studentadditionalND$class)
print("Confusion Matrix - Test Data (Cost-Complexity Pruning):")
print(classificationtable_test_new)
acctesttree_test_new <- sum(diag(classificationtable_test_new)) / sum(classificationtable_test_new)
cat("Test Accuracy (Cost-Complexity Pruning):", round(acctesttree_test_new * 100, 2), "%\n")

train_accuracy_tree_costpruned <- acctesttree_training_new
test_accuracy__tree_costpruned <- acctesttree_test_new





# Part 1 iv - Random Forrest 
library(randomForest)
library(ggplot2)
set.seed(2024)

myrf_first_first <- randomForest(class ~ ., studentND, ntree=1000, mtry=4, importance=TRUE)

importance(myrf_first_first)
varImpPlot(myrf_first_first, main = "Variable Importance Plot for the Random Forest Model")

myprediction <- predict(myrf_first_first, studentadditionalND, type='class')
classificationtable <- table(myprediction, studentadditionalND$class)
print(classificationtable)
acctestrandomforest_base <- sum(diag(classificationtable)) / sum(classificationtable)
acctestrandomforest_base

myprediction <- predict(myrf_first_first, studentND, type='class')
classificationtable <- table(myprediction, studentND$class)
print(classificationtable)

acctrainingrandomforest_base <- sum(diag(classificationtable)) / sum(classificationtable)
acctrainingrandomforest_base

train_accuracy_rf_baseline <- acctrainingrandomforest_base
test_accuracy__rf_baseline <- acctestrandomforest_base

# Parameter grids
ntree_values <- c(100, 150 , 200, 300, 400, 500, 1000, 1500, 2000)
mtry_values <- c(1, 2, 3, 4, 5, 6, 7, 8, 9)

# Results dataframe
results <- data.frame(ntree = integer(), mtry = integer(), OOB_Error = numeric())

# Nested loop for tuning
for (ntree in ntree_values) {
  for (mtry in mtry_values) {
    set.seed(2024)  
    rf_model <- randomForest(class ~ ., data = studentND, ntree = ntree, mtry = mtry, importance = TRUE)
    OOB_Error <- rf_model$err.rate[ntree, "OOB"]
    results <- rbind(results, data.frame(ntree = ntree, mtry = mtry, OOB_Error = OOB_Error))
  }
}

# Find best combination
best_combination <- results[which.min(results$OOB_Error), ]
print(results)
print(paste("Bedste kombination: ntree =", best_combination$ntree, ", mtry =", best_combination$mtry))
print(paste("Lowest OOB error:", best_combination$OOB_Error))

# Create Model column for ggplot
results$Model <- paste("ntree", results$ntree, "mtry", results$mtry, sep = "_")

# Filter results to include only rows with OOB_Error <= 0.31
filtered_results <- results[results$OOB_Error <= 0.31, ]

# Plot with ggplot2 using the filtered results
ggplot(filtered_results, aes(x = Model, y = OOB_Error)) +
  geom_point(color = "blue") +
  geom_text(aes(label = round(OOB_Error, 4)), vjust = -0.5) +
  labs(title = "OOB Error Comparison for Models with OOB Error <= 0.31",
       x = "Model (ntree_mtry)",
       y = "OOB Error") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

set.seed(2024)
final_random_forrest_model <- randomForest(class ~ ., studentND, ntree=100, mtry=8, importance=TRUE)

set.seed(2024)
myprediction_optimal_test <- predict(final_random_forrest_model, studentadditionalND, type='class')
classificationtable_optimal_test <- table(myprediction_optimal_test, studentadditionalND$class)
print(classificationtable_optimal_test)
acctestrandomforest_optimal <- sum(diag(classificationtable_optimal_test)) / sum(classificationtable_optimal_test)
acctestrandomforest_optimal

myprediction <- predict(myrf_first, studentND, type='class')
classificationtable <- table(myprediction, studentND$class)
print(classificationtable)

acctrainingrandomforest <- sum(diag(classificationtable)) / sum(classificationtable)
acctrainingrandomforest

train_accuracy_rf_optimal <- acctrainingrandomforest
train_accuracy_rf_optimal
test_accuracy__rf_optimal <- acctestrandomforest_optimal
test_accuracy__rf_optimal


# Part 1 v k-Nearest Neighbors
set.seed(2024)
library(FNN)
myxtrain <- studentND[,-which(names(studentND) == "class")]
myytrain <- studentND$class
myxtesting <- studentadditionalND[,-which(names(studentadditionalND) == "class")]
myytesting <- studentadditionalND$class

# K-NN K=1
myk1nn <- knn(train=myxtrain, test=myxtesting, cl=myytrain, k=1)
# Beregn træningsnøjagtighed for k-NN med k=1
myk1nn_train <- knn(train = myxtrain, test = myxtrain, cl = myytrain, k = 1)
myaccuracytablek1nn_train <- table(myk1nn_train, myytrain)
mytrainingaccuracyk1nn <- sum(diag(myaccuracytablek1nn_train)) / sum(myaccuracytablek1nn_train)
print(paste("Træningsnøjagtighed for K=1:", round(mytrainingaccuracyk1nn * 100, 2), "%"))


myaccuracytablek1nn <- table(myk1nn, myytesting)
mytestingaccuracyk1nn <- sum(diag(myaccuracytablek1nn)) / sum(myaccuracytablek1nn)
print(paste("Test Accuracy for K=1:", round(mytestingaccuracyk1nn * 100, 2), "%"))

# K-NN K=3
myk3nn <- knn(train=myxtrain, test=myxtesting, cl=myytrain, k=3)

myk3nn_train <- knn(train = myxtrain, test = myxtrain, cl = myytrain, k = 3)
myaccuracytablek3nn_train <- table(myk3nn_train, myytrain)
mytrainingaccuracyk3nn <- sum(diag(myaccuracytablek3nn_train)) / sum(myaccuracytablek3nn_train)
myaccuracytablek3nn_train
print(paste("Træningsnøjagtighed for K=3:", round(mytrainingaccuracyk3nn * 100, 2), "%"))

myaccuracytablek3nn <- table(myk3nn, myytesting)
mytestingaccuracyk3nn <- sum(diag(myaccuracytablek3nn)) / sum(myaccuracytablek3nn)
print(paste("Test Accuracy for K=3:", round(mytestingaccuracyk3nn * 100, 2), "%"))

# Cross-Validation
bestk <- 0
bestaccuracy <- 0
accuracy <- NULL

set.seed(2024)
for(auxk in 1:15) {
  mycv <- knn.cv(train=myxtrain, cl=myytrain, k=auxk)
  mytable <- table(mycv, myytrain)
  accuracy[auxk] <- sum(diag(mytable)) / sum(mytable)
  
  if(bestaccuracy < accuracy[auxk]) {
    bestk <- auxk
    bestaccuracy <- accuracy[auxk]
  }
}
print(bestaccuracy)
plot(accuracy, type = "b", xlab="K", ylab="Cross-validated Accuracy", main="Cross-validated Accuracy for Different K")
print(paste("Best K:", bestk))
accuracy[1]
mybestknn <- knn(train=myxtrain, test=myxtesting, cl=myytrain, k=bestk)

mytestingaccuracytablebestknn <- table(mybestknn, myytesting)
mytestingaccuracybestknn <- sum(diag(mytestingaccuracytablebestknn)) / sum(mytestingaccuracytablebestknn)
print(paste("Test Accuracy with Best K=", bestk, ":", round(mytestingaccuracybestknn * 100, 2), "%"))
mytestingaccuracytablebestknn

train_accuracy_knn_3 <- mytrainingaccuracyk3nn
test_accuracy__knn_3 <- mytestingaccuracybestknn

# Part 1 vi ny #
train_accuracy_logistic
test_accuracy_logistic 
train_accuracy_svm_linear 
test_accuracy__svm_linear
train_accuracy_svm_polynomial 
test_accuracy__svm_polynomial 
train_accuracy_svm_radial 
test_accuracy__svm_radial 
train_accuracy_svm_sigmoid 
test_accuracy__svm_sigmoid
train_accuracy_tree_standard
test_accuracy__tree_standard
train_accuracy_tree_sizedpruned
test_accuracy__tree_sizedpruned
train_accuracy_tree_costpruned
test_accuracy__tree_costpruned
train_accuracy_rf_baseline
test_accuracy__rf_baseline
train_accuracy_rf_optimal 
test_accuracy__rf_optimal
train_accuracy_knn_3
test_accuracy__knn_3

# Load necessary libraries
library(e1071)        # For SVM testing
library(randomForest) # For Random Forest testing
library(FNN)          # For k-NN testing
library(ggplot2)      # For plotting

# Set seed for reproducibility
set.seed(2024)

# Initialize data frames for storing results
model_results_train <- data.frame(Model = character(),
                                  TrainAccuracy = numeric(),
                                  stringsAsFactors = FALSE)

model_results_test <- data.frame(Model = character(),
                                 TestAccuracy = numeric(),
                                 stringsAsFactors = FALSE)
set.seed(2024)
# Logistic Regression - Training Accuracy
train_predictions_logistic <- predict(stepwise_model, newdata = studentND, type = "response")
train_pred_class_logistic <- ifelse(train_predictions_logistic > 0.5, "Low", "High")
train_accuracy_logistic <- mean(studentND$class == train_pred_class_logistic)
model_results_train <- rbind(model_results_train, data.frame(Model = "Logistic Regression", TrainAccuracy = train_accuracy_logistic))
set.seed(2024)
# Logistic Regression - Test Accuracy
test_predictions_logistic <- predict(stepwise_model, newdata = studentadditionalND, type = "response")
test_pred_class_logistic <- ifelse(test_predictions_logistic > 0.5, "Low", "High")
test_accuracy_logistic <- mean(studentadditionalND$class == test_pred_class_logistic)
model_results_test <- rbind(model_results_test, data.frame(Model = "Logistic Regression", TestAccuracy = test_accuracy_logistic))

set.seed(2024)
# SVM Models - Training and Test Accuracy
kernels <- list("Linear" = best_model_linear, "Polynomial" = best_model_polynomial, "Radial" = best_model_radial, "Sigmoid" = best_model_sigmoid)

for (kernel_name in names(kernels)) {
  # Training Accuracy
  train_accuracy <- sum(diag(table(predict(kernels[[kernel_name]], studentND), studentND$class))) / nrow(studentND)
  model_results_train <- rbind(model_results_train, data.frame(Model = paste(kernel_name, "SVM"), TrainAccuracy = train_accuracy))
  
  # Test Accuracy
  test_accuracy <- sum(diag(table(predict(kernels[[kernel_name]], studentadditionalND), studentadditionalND$class))) / nrow(studentadditionalND)
  model_results_test <- rbind(model_results_test, data.frame(Model = paste(kernel_name, "SVM"), TestAccuracy = test_accuracy))
}
set.seed(2024)
# Classification Tree (Unpruned) - Training and Test Accuracy
train_predictions_tree_unpruned <- predict(mytree, studentND, type='class')
train_accuracy_tree_unpruned <- mean(studentND$class == train_predictions_tree_unpruned)
model_results_train <- rbind(model_results_train, data.frame(Model = "Classification Tree (Unpruned)", TrainAccuracy = train_accuracy_tree_unpruned))

test_predictions_tree_unpruned <- predict(mytree, studentadditionalND, type='class')
test_accuracy_tree_unpruned <- mean(studentadditionalND$class == test_predictions_tree_unpruned)
model_results_test <- rbind(model_results_test, data.frame(Model = "Classification Tree (Unpruned)", TestAccuracy = test_accuracy_tree_unpruned))

set.seed(2024)
# Classification Tree (Pruned) - Training and Test Accuracy
train_predictions_tree_pruned <- predict(myprunedtree_new, studentND, type='class')
train_accuracy_tree_pruned <- mean(studentND$class == train_predictions_tree_pruned)
model_results_train <- rbind(model_results_train, data.frame(Model = "Classification Tree (Pruned)", TrainAccuracy = train_accuracy_tree_pruned))

test_predictions_tree_pruned <- predict(myprunedtree_new, studentadditionalND, type='class')
test_accuracy_tree_pruned <- mean(studentadditionalND$class == test_predictions_tree_pruned)
model_results_test <- rbind(model_results_test, data.frame(Model = "Classification Tree (Pruned)", TestAccuracy = test_accuracy_tree_pruned))

set.seed(2024)
# Random Forest - Training and Test Accuracy

final_random_forrest_model <- randomForest(class ~ ., studentND, ntree=100, mtry=8, importance=TRUE)

set.seed(2024)
myprediction_test <- predict(final_random_forrest_model, studentadditionalND, type='class')
classificationtable_test <- table(myprediction_test, studentadditionalND$class)
print(classificationtable_test)
acctestrandomforest <- sum(diag(classificationtable_test)) / sum(classificationtable_test)
acctestrandomforest

set.seed(2024)
myprediction <- predict(final_random_forrest_model, studentND, type='class')
classificationtable_training_rf <- table(myprediction, studentND$class)
print(classificationtable_training_rf)
acctrainingrandomforest <- sum(diag(classificationtable_training_rf)) / sum(classificationtable_training_rf)
model_results_test <- rbind(model_results_test, data.frame(Model = "Random Forest", TestAccuracy = acctestrandomforest))


train_accuracy_rf <- sum(diag(classificationtable_training_rf)) / sum(classificationtable_training_rf)
train_accuracy_rf

model_results_train <- rbind(model_results_train, data.frame(Model = "Random Forest", TrainAccuracy = train_accuracy_rf))


# k-Nearest Neighbors with best K (identified from cross-validation)
mybestknn_train <- knn(train=myxtrain, test=myxtrain, cl=myytrain, k=bestk)
train_accuracy_knn <- mean(myytrain == mybestknn_train)
model_results_train <- rbind(model_results_train, data.frame(Model = paste("k-NN (K =", bestk, ")"), TrainAccuracy = train_accuracy_knn))

mybestknn_test <- knn(train=myxtrain, test=myxtesting, cl=myytrain, k=bestk)
test_accuracy_knn <- mean(myytesting == mybestknn_test)
model_results_test <- rbind(model_results_test, data.frame(Model = paste("k-NN (K =", bestk, ")"), TestAccuracy = test_accuracy_knn))

# Merge results and reshape for plotting
model_comparison <- merge(model_results_train, model_results_test, by = "Model", all = TRUE)
print(model_comparison)
# Plot comparison of Train and Test Accuracy
library(reshape2)
model_comparison_long <- melt(model_comparison, id.vars = "Model", variable.name = "Dataset", value.name = "Accuracy")

# Plot for Training Accuracy
ggplot(model_comparison, aes(x = reorder(Model, -TrainAccuracy), y = TrainAccuracy * 100)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  geom_text(aes(label = paste0(round(TrainAccuracy * 100, 2), "%")), vjust = -0.5) +
  labs(title = "Training Accuracy for Different Models",
       x = "Model",
       y = "Training Accuracy (%)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Plot for Test Accuracy
ggplot(model_comparison, aes(x = reorder(Model, -TestAccuracy), y = TestAccuracy * 100)) +
  geom_bar(stat = "identity", fill = "coral") +
  geom_text(aes(label = paste0(round(TestAccuracy * 100, 2), "%")), vjust = -0.5) +
  labs(title = "Test Accuracy for Different Models",
       x = "Model",
       y = "Test Accuracy (%)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))









#### Part 1 vi ####

# Load necessary libraries
library(e1071)       # For SVM testing
library(randomForest) # For Random Forest testing
library(FNN)         # For k-NN testing
library(ggplot2)     # For plotting

# Set seed for reproducibility
set.seed(2024)

# Initialize data frame for storing results
model_results <- data.frame(Model = character(),
                            TestAccuracy = numeric(),
                            stringsAsFactors = FALSE)

# Logistic Regression Test
test_predictions_logistic <- predict(stepwise_model, newdata = studentadditionalND, type = "response")
test_pred_class_logistic <- ifelse(test_predictions_logistic > 0.5, "Low", "High")
test_conf_matrix_logistic <- table(studentadditionalND$class, test_pred_class_logistic)
test_accuracy_logistic <- mean(studentadditionalND$class == test_pred_class_logistic)
model_results <- rbind(model_results, data.frame(Model = "Logistic Regression", TestAccuracy = test_accuracy_logistic))


# SVM Models Test
# Linear Kernel
test_accuracy_linear <- sum(diag(table(predict(best_model_linear, studentadditionalND), 
                                       studentadditionalND$class))) / nrow(studentadditionalND)
model_results <- rbind(model_results, data.frame(Model = "Linear SVM", TestAccuracy = test_accuracy_linear))

# Polynomial Kernel
test_accuracy_polynomial <- sum(diag(table(predict(best_model_polynomial, studentadditionalND), 
                                           studentadditionalND$class))) / nrow(studentadditionalND)
model_results <- rbind(model_results, data.frame(Model = "Polynomial SVM", TestAccuracy = test_accuracy_polynomial))

# Radial Kernel
test_accuracy_radial <- sum(diag(table(predict(best_model_radial, studentadditionalND), 
                                       studentadditionalND$class))) / nrow(studentadditionalND)
model_results <- rbind(model_results, data.frame(Model = "Radial SVM", TestAccuracy = test_accuracy_radial))

# Classification Tree Test (Unpruned)
set.seed(2024)
myprediction_tree_unpruned <- predict(mytree, studentadditionalND, type='class')
classificationtable_tree_unpruned <- table(myprediction_tree_unpruned, studentadditionalND$class)
acctest_tree_unpruned <- sum(diag(classificationtable_tree_unpruned)) / sum(classificationtable_tree_unpruned)
model_results <- rbind(model_results, data.frame(Model = "Classification Tree (Unpruned)", TestAccuracy = acctest_tree_unpruned))

# Classification Tree Test (Pruned)
myprediction_tree_pruned_cost <- predict(myprunedtree_new_cost, studentadditionalND, type='class')
classificationtable_tree_pruned_cost <- table(myprediction_tree_pruned_cost, studentadditionalND$class)
acctest_tree_pruned <- sum(diag(classificationtable_tree_pruned_cost)) / sum(classificationtable_tree_pruned_cost)
model_results <- rbind(model_results, data.frame(Model = "Classification Tree (Cost)", TestAccuracy = acctest_tree_pruned))

# Random Forest Test ntree=100, mtry=8
rf_test_predictions <- predict(final_random_forrest_model, studentadditionalND, type = 'class')
rf_test_conf_matrix <- table(rf_test_predictions, studentadditionalND$class)
rf_test_accuracy <- sum(diag(rf_test_conf_matrix)) / sum(rf_test_conf_matrix)
model_results <- rbind(model_results, data.frame(Model = "Random Forest", TestAccuracy = rf_test_accuracy))

# k-Nearest Neighbors with best K (identified from cross-validation)
myxtrain <- studentND[,-which(names(studentND) == "class")]
myytrain <- studentND$class
myxtesting <- studentadditionalND[,-which(names(studentadditionalND) == "class")]
myytesting <- studentadditionalND$class
mybestknn <- knn(train=myxtrain, test=myxtesting, cl=myytrain, k=bestk)
mytestingaccuracytable_bestknn <- table(mybestknn, myytesting)
mytestingaccuracy_bestknn <- sum(diag(mytestingaccuracytable_bestknn)) / sum(mytestingaccuracytable_bestknn)
model_results <- rbind(model_results, data.frame(Model = paste("k-NN (K =", bestk, ")"), TestAccuracy = mytestingaccuracy_bestknn))

# Plot the Test Accuracy Comparison
ggplot(model_results, aes(x = reorder(Model, -TestAccuracy), y = TestAccuracy * 100)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  geom_text(aes(label = paste0(round(TestAccuracy * 100, 2), "%")), vjust = -0.5) +
  labs(title = "Model Comparison on Test Data",
       x = "Model",
       y = "Test Accuracy (%)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# table
class_counts1 <- table(studentadditional$class)

bar_positions <- barplot(class_counts1, 
                         main = "Frequency of High and Low Categories", 
                         col = c("lightblue", "lightcoral"),
                         ylab = "Frequency",
                         xlab = "Category",
                         ylim = c(0, max(class_counts1) + 10)) 

text(x = bar_positions, 
     y = class_counts1, 
     labels = class_counts1, 
     pos = 3, 
     cex = 0.8, 
     col = "black")

#### Part 2 ####


#### Data Processing ####

#standardizes
wine <- read.table("wine.txt", header=TRUE, stringsAsFactors = TRUE)
summary(wine)
means <- apply(wine, 2, mean)
standarddeviations <- apply(wine, 2, sd)
wineSTAN <- scale(wine, center=means, scale=standarddeviations)
summary(wineSTAN)
var(wineSTAN)


#normalize

normalize <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}
wineNORM <- as.data.frame(lapply(wine, normalize))
summary(wineNORM)


# Part 2 i
#standardizes

mydistmatrix <- dist(wineSTAN)

mydistsubmatrix <- as.matrix(mydistmatrix)
mydistsubmatrix <- mydistsubmatrix[1:10, 1:10]
print(mydistsubmatrix)

# Hierarkisk clustering with complete linkage
myahclust_complete <- hclust(mydistmatrix, method = "complete")
plot(myahclust_complete, hang = -1, main = "Dendrogram for Wine Data (Complete Linkage - Euclidean Distance)", 
     xlab = "Sample Index", ylab = "Euclidean Distance", cex = 0.75)
rect.hclust(myahclust_complete, k = 3, border = "red")
clusters <- cutree(myahclust_complete, h = 8.72)  
table(clusters)


# Appendix



# Hierarkisk clustering with single linkage
myahclust_single <- hclust(mydistmatrix, method = "single")
plot(myahclust_single, hang = -1, main = "Dendrogram for Wine Data (Single Linkage - Euclidean Distance)", 
     xlab = "Sample Index", ylab = "Euclidean Distance", cex = 0.75)

# Hierarkisk clustering with average linkage
myahclust_average <- hclust(mydistmatrix, method = "average")
plot(myahclust_average, hang = -1, main = "Dendrogram for Wine Data (Average Linkage - Euclidean Distance)", 
     xlab = "Sample Index", ylab = "Euclidean Distance", cex = 0.75)
#rect.hclust(myahclust_average, k = 7, border = "red")
# Compute additional distance matrices using different distance metrics
mydistmatrix_manhattan <- dist(wineSTAN, method = "manhattan")
mydistmatrix_maximum <- dist(wineSTAN, method = "maximum")

# Hierarchical clustering with complete linkage and Manhattan distance
myahclust_complete_manhattan <- hclust(mydistmatrix_manhattan, method = "complete")
plot(myahclust_complete_manhattan, hang = -1, 
     main = "Dendrogram for Wine Data (Complete Linkage - Manhattan Distance)", 
     xlab = "Sample Index", ylab = "Manhattan Distance", cex = 0.75)
rect.hclust(myahclust_complete_manhattan, k = 2, border = "blue")

# Hierarchical clustering with single linkage and Manhattan distance
myahclust_single_manhattan <- hclust(mydistmatrix_manhattan, method = "single")
plot(myahclust_single_manhattan, hang = -1, 
     main = "Dendrogram for Wine Data (Single Linkage - Manhattan Distance)", 
     xlab = "Sample Index", ylab = "Manhattan Distance", cex = 0.75)

# Hierarchical clustering with average linkage and Manhattan distance
myahclust_average_manhattan <- hclust(mydistmatrix_manhattan, method = "average")
plot(myahclust_average_manhattan, hang = -1, 
     main = "Dendrogram for Wine Data (Average Linkage - Manhattan Distance)", 
     xlab = "Sample Index", ylab = "Manhattan Distance", cex = 0.75)

# Hierarchical clustering with complete linkage and Maximum distance
myahclust_complete_maximum <- hclust(mydistmatrix_maximum, method = "complete")
plot(myahclust_complete_maximum, hang = -1, 
     main = "Dendrogram for Wine Data (Complete Linkage - Maximum Distance)", 
     xlab = "Sample Index", ylab = "Maximum Distance", cex = 0.75)
rect.hclust(myahclust_complete_maximum, k = 2, border = "green")

# Hierarchical clustering with single linkage and Maximum distance
myahclust_single_maximum <- hclust(mydistmatrix_maximum, method = "single")
plot(myahclust_single_maximum, hang = -1, 
     main = "Dendrogram for Wine Data (Single Linkage - Maximum Distance)", 
     xlab = "Sample Index", ylab = "Maximum Distance", cex = 0.75)

# Hierarchical clustering with average linkage and Maximum distance
myahclust_average_maximum <- hclust(mydistmatrix_maximum, method = "average")
plot(myahclust_average_maximum, hang = -1, 
     main = "Dendrogram for Wine Data (Average Linkage - Maximum Distance)", 
     xlab = "Sample Index", ylab = "Maximum Distance", cex = 0.75)


# Part 2 ii
myKmeans3 <- kmeans(wineSTAN, 3)

with(as.data.frame(wineSTAN), pairs(as.data.frame(wineSTAN)[,1:2], col=c(1:11)[myKmeans3$cluster]))

# Cluster Size
myKmeans3$size

# Sum of squares within each cluster and total
myKmeans3$withinss 
myKmeans3$tot.withinss

# Kør K-Means clustering med en anden seed
set.seed(2025)
myKmeans3 <- kmeans(wineSTAN, 3)
myKmeans3$tot.withinss
myKmeans3 <- kmeans(wineSTAN, 3)

myKmeans3$tot.withinss

# Run K-Means clustering with nstart=50 to achieve the best result
set.seed(2024)
myKmeans3multistart <- kmeans(wineSTAN, 3, nstart=50)
myKmeans3multistart$tot.withinss
mytotwithinss <- NULL

for(auxk in 1:15) {
  set.seed(2024)
  myKmeansauxkmultistart <- kmeans(wineSTAN, auxk, nstart=50)
  mytotwithinss[auxk] <- myKmeansauxkmultistart$tot.withinss
}

plot(1:15, mytotwithinss[1:15], type = "b", xlab = 'Number of Clusters (k)', ylab = 'Total Within Sum of Squares', main = "Elbow Method for Optimal k")



# Part 2i
# Normalize
mydistmatrix <- dist(wineNORM)

mydistsubmatrix <- as.matrix(mydistmatrix)
mydistsubmatrix <- mydistsubmatrix[1:10, 1:10]
print(mydistsubmatrix)

myahclust_single <- hclust(mydistmatrix, method = "single")
plot(myahclust_single, hang = -1, main = "Dendrogram for Wine Data (Single Linkage - Euclidean Distance)", 
     xlab = "Sample Index", ylab = "Euclidean Distance", cex = 0.75)

myahclust_complete <- hclust(mydistmatrix, method = "complete")
plot(myahclust_complete, hang = -1, main = "Dendrogram for Wine Data (Complete Linkage - Euclidean Distance)", 
     xlab = "Sample Index", ylab = "Euclidean Distance", cex = 0.75)

clusters <- cutree(myahclust_complete, h = 6)
table(clusters)  

myahclust_average <- hclust(mydistmatrix, method = "average")
plot(myahclust_average, hang = -1, main = "Dendrogram for Wine Data (Average Linkage - Euclidean Distance)", 
     xlab = "Sample Index", ylab = "Euclidean Distance", cex = 0.75)

# Part 2 ii
myKmeans3 <- kmeans(wineNORM, 3)

with(as.data.frame(wineNORM), pairs(as.data.frame(wineNORM)[,1:2], col=c(1:11)[myKmeans3$cluster]))

myKmeans3$size
myKmeans3$withinss 

myKmeans3$tot.withinss

set.seed(2025)
myKmeans3 <- kmeans(wineNORM, 3)
myKmeans3$tot.withinss

set.seed(2024)
myKmeans3multistart <- kmeans(wineNORM, 3, nstart=50)
myKmeans3multistart$tot.withinss
mytotwithinss <- NULL

for(auxk in 1:15) {
  set.seed(2024)
  myKmeansauxkmultistart <- kmeans(wineNORM, auxk, nstart=50)
  mytotwithinss[auxk] <- myKmeansauxkmultistart$tot.withinss
}

plot(1:15, mytotwithinss[1:15], type = "b", xlab = 'Number of Clusters (k)', ylab = 'Total Within Sum of Squares', main = "Elbow Method for Optimal k")





#### Part 3 ####

#Data Processing

seeds <- read.table("seeds.txt", header=TRUE, stringsAsFactors = TRUE)
summary(seeds)
means <- apply(seeds, 2, mean)
standarddeviations <- apply(seeds, 2, sd)
seedsSTAN <- scale(seeds, center=means, scale=standarddeviations)
summary(seedsSTAN)
row.names(seedsSTAN) <- row.names(seeds)



# Part 3 i
covmatseeds <- cov(seeds)
print(round(covmatseeds, 2))

myPCA <- prcomp(seeds, center=TRUE, scale.=TRUE)

x <- myPCA$x[,1]
y <- myPCA$x[,2]

par(mar = c(5, 5, 4, 2) + 0.1)  

plot(x, y, xlab="Principal Component 1", ylab="Principal Component 2", 
     main="PCA of Seeds Data", pch=16, col="blue", cex=1.2, xlim=c(-3, 5), ylim=c(-3, 3))
grid()  

text(x, y, labels=row.names(seeds), pos=4, cex=0.7, col="darkred")

library('scatterplot3d')

x <- myPCA$x[,1]
y <- myPCA$x[,2]
z <- myPCA$x[,3]

scatterplot3d(x, y, z, xlab="Principal Component 1", ylab="Principal Component 2", 
              zlab="Principal Component 3", main="PCA of Seeds Data", pch=16, color="blue", 
              cex.symbols=1.1, grid=TRUE, angle=40)  

text(x, y, z, labels=row.names(seeds), pos=4, cex=0.6, col="darkred")

covmatPCA <- cov(myPCA$x)
sum(diag(covmatPCA))

summary(myPCA)


# appendices plots
variances <- myPCA$sdev^2
explained_variance <- variances / sum(variances) * 100
plot(explained_variance, type="b", xlab="Principal Component", 
     ylab="Explained Variance (%)", main="Scree Plot of Explained Variance")

# Biplot PCA
biplot(myPCA, scale=0, cex=0.6, col=c("grey", "blue"))

pairs(seedsSTAN, main="Pairwise Scatter Plots of Standardized Variables", pch=16, col="blue")

par(mar = c(5, 5, 4, 2) + 0.1)  

variances <- myPCA$sdev^2
explained_variance <- variances / sum(variances)
cumulative_variance <- cumsum(explained_variance) * 100  

plot(cumulative_variance, type="b", xlab="Principal Component", 
     ylab="Cumulative Explained Variance (%)", main="Cumulative Scree Plot", 
     pch=16, col="blue")

abline(h=80, col="red", lty=2)  
abline(h=90, col="green", lty=2)  


## Part 3 ii

pairs(seedsSTAN, main = "Pairwise Scatter Plots of Standardized Variables", pch = 16, col = "blue")

# ---- MDS with Euclidean Distance ----
myMDS_euclidean <- cmdscale(dist(seedsSTAN), 2, eig = TRUE)
x <- myMDS_euclidean$points[, 1]
y <- myMDS_euclidean$points[, 2]
plot(x, y, xlab = "Coordinate 1", ylab = "Coordinate 2", 
     main = "MDS (Euclidean Distance)", pch = 16, col = "blue", cex = 1.2)
grid()
text(x, y, labels = row.names(seedsSTAN), pos = 4, cex = 0.7, col = "darkred")

# ---- MDS with Manhattan Distance ----
myMDS_manhattan <- cmdscale(dist(seedsSTAN, method = "manhattan"), 2, eig = TRUE)
x <- myMDS_manhattan$points[, 1]
y <- myMDS_manhattan$points[, 2]
plot(x, y, xlab = "Coordinate 1", ylab = "Coordinate 2", 
     main = "MDS (Manhattan Distance)", pch = 16, col = "blue", cex = 1.2)
grid()
text(x, y, labels = row.names(seedsSTAN), pos = 4, cex = 0.7, col = "darkred")

# ---- MDS with Maximum Distance ----
myMDS_maximum <- cmdscale(dist(seedsSTAN, method = "maximum"), 2, eig = TRUE)
x <- myMDS_maximum$points[, 1]
y <- myMDS_maximum$points[, 2]
plot(x, y, xlab = "Coordinate 1", ylab = "Coordinate 2", 
     main = "MDS (Maximum Distance)", pch = 16, col = "blue", cex = 1.2)
grid()
text(x, y, labels = row.names(seedsSTAN), pos = 4, cex = 0.7, col = "darkred")

# Check if the row names are correctly set
print(row.names(seedsSTAN))


print(row.names(seeds))




# Part ii
# Normalize
minimum <- apply(seeds, 2, min)
maximum <- apply(seeds, 2, max)
seedsNORM <- scale(seeds, center = minimum, scale = (maximum - minimum))
myseedsNORM <- as.data.frame(seedsNORM)

# Summary and pairwise plot
summary(myseedsNORM)
with(myseedsNORM, pairs(myseedsNORM))

# ---- MDS with Euclidean Distance ----
myMDS <- cmdscale(dist(myseedsNORM), 2, eig = TRUE)
x <- myMDS$points[, 1]
y <- myMDS$points[, 2]
plot(x, y, xlab = "Coordinate 1", ylab = "Coordinate 2", main = "MDS (Euclidean Distance)")
text(x, y, labels = row.names(myseedsNORM), cex = 0.7)

# ---- MDS with Manhattan Distance ----
myMDSManhattan <- cmdscale(dist(myseedsNORM, method = "manhattan"), 2, eig = TRUE)
x <- myMDSManhattan$points[, 1]
y <- myMDSManhattan$points[, 2]
plot(x, y, xlab = "Coordinate 1", ylab = "Coordinate 2", main = "MDS (Manhattan Distance)")
text(x, y, labels = row.names(myseedsNORM), cex = 0.7)

# ---- MDS with Maximum Distance ----
myMDSMaximum <- cmdscale(dist(myseedsNORM, method = "maximum"), 2, eig = TRUE)
x <- myMDSMaximum$points[, 1]
y <- myMDSMaximum$points[, 2]

# Plot MDS results with enhanced visualization
plot(x, y, xlab = "Coordinate 1", ylab = "Coordinate 2", main = "MDS (Maximum Distance)", 
     pch = 16, col = "blue", cex = 1.2)
grid()
text(x, y, labels = row.names(myseedsNORM), pos = 4, cex = 0.7, col = "darkred")
