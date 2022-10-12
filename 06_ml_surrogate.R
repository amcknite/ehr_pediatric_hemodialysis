## Creates surrogate model based on random forest

## Load libraries
set.seed(1234)
library(tidyverse)
library(mlr3)
library(mlr3learners)
library(mlr3viz)
library(mlr3tuning)
library(paradox)
library(pdp)
library(vip)
library(iml)

## Load data
load("data_for_ml.RData")

## Set all features as factors and remove columns (features) not used for model
final_dat <- final_dat %>% 
  mutate(code = as.factor(code_b),
         sex = as.numeric(sex == "F")) %>%
  select(-patient_id, -encounter_id, -code_b, -race, -ethnicity)

## ----------------------------------------------------------------------------
##MLR3 setup
## Set up task-outline of what models will do
task_drugs <- TaskClassif$new(id = "drugs", backend = final_dat, 
                              target = "code")

task_drugs$col_roles$stratum <- "patient_id"

## ----------------------------------------------------------------------------
## Randome Forest model
lrn_rf_vi <- lrn("classif.ranger", predict_type = "prob",
                 mtry.ratio = 0.222222,
                 sample.fraction = 0.9,
                 num.trees = 450,
                 importance = "permutation")

lrn_rf_vi$train(task_drugs)

## ----------------------------------------------------------------------------
## Repredict for surrogate model
lrn_rf_pred <- lrn_rf_vi$predict_newdata(final_dat_test)
lrn_rf_pred <- lrn_rf_vi$predict(task_drugs)

## ----------------------------------------------------------------------------
## Uses RPART to create surrogate
library(rpart)
library(rpart.plot)

## Replace final column with predicted modality (CRRT or iHD)
x <- final_dat[, -231]
surrogate_data <- cbind(x, lrn_rf_pred$data$response)
names(surrogate_data)[ncol(surrogate_data)] <- "code"

## Fit model 
lrn_rpart <- rpart(code ~., surrogate_data, 
                   control = rpart.control(maxdepth = 3))
lrn_rpart
prp(lrn_rpart)

## Cross-validation of surrogate using MLR3
task_surrogate <- TaskClassif$new(id = "surrogate", 
                                  backend = surrogate_data, 
                                  target = "code")

## Define performance metrics
measure_auc <- msr("classif.auc")
measure_sen <- msr("classif.sensitivity")
measure_spe <- msr("classif.specificity")
measure_acc <- msr("classif.acc")
measure_bacc <- msr("classif.bacc")

## Define repeated k-fold cross validation
cv_rsmp <- rsmp("repeated_cv", folds = 5, repeats = 10)
cv_rsmp$instantiate(task_drugs)


## Define a learner (algorithm) - surrogate model
lrn_rpart <- lrn("classif.rpart", predict_type = "prob")
rr_rpart <- resample(task_surrogate, lrn_rpart, cv_rsmp)

rr_rpart$score(measure_auc)
rr_rpart$aggregate(measure_auc)
rr_rpart$aggregate(measure_sen)
rr_rpart$aggregate(measure_spe)
rr_rpart$aggregate(measure_acc)
rr_rpart$aggregate(measure_bacc)
