set.seed(1234)
library(tidyverse)
library(mlr3)
library(mlr3learners)
# library(mlr3measures)
library(mlr3viz)
library(mlr3tuning)
library(paradox)
library(pdp)
library(vip)
library(iml)

load("data_for_ml.RData")

y_train <- final_dat %>% 
  mutate(code = as.factor(code_b),
         sex = as.numeric(sex == "F")) %>%
  select(code_b)

final_dat <- final_dat %>% 
  mutate(code = as.factor(code_b),
         sex = as.numeric(sex == "F")) %>%
  select(-patient_id, -encounter_id, -code_b, -race, -ethnicity)

load("ih_data_for_prediction.RData")

y_test <- ih_out %>% 
  mutate(code = as.factor(code_b),
         sex = as.numeric(sex == "F")) %>%
  select(code_b)

final_dat_test <- ih_out %>% 
  mutate(code = as.factor(code_b),
         sex = as.numeric(sex == "F")) %>%
  select(-patient_id, -encounter_id, -code_b)

## ----------------------------------------------------------------------------
##MLR3 setup
## Set up task
task_drugs <- TaskClassif$new(id = "drugs", backend = final_dat, 
                              target = "code")

task_drugs$col_roles$stratum <- "code"

## ----------------------------------------------------------------------------
# helper function to try different threshold values interactively
cost_measure = msr("classif.bacc")
with_threshold = function(p, th) {
  p$set_threshold(th)
  list(confusion = p$confusion, costs = p$score(measures = cost_measure, 
                                                task = task_drugs))
}

## ----------------------------------------------------------------------------
## Ranger model
lrn_rf_vi <- lrn("classif.ranger", predict_type = "prob",
                 mtry.ratio = 0.222222,
                 sample.fraction = 0.9,
                 num.trees = 450,
                 importance = "permutation")

lrn_rf_vi$train(task_drugs)

## ----------------------------------------------------------------------------
##MLR3 setup
lrn_rf_pred <- lrn_rf_vi$predict_newdata(final_dat_test)

lrn_rf_pred <- lrn_rf_vi$predict(task_drugs)

library(rpart)
library(rpart.plot)
x <- final_dat[, -231]
surrogate_data <- cbind(x, lrn_rf_pred$data$response)
names(surrogate_data)[ncol(surrogate_data)] <- "code"
lrn_rpart <- rpart(code ~., surrogate_data, 
                   control = rpart.control(maxdepth = 3))
lrn_rpart
prp(lrn_rpart)

## Cross validation of surrogate
task_surrogate <- TaskClassif$new(id = "surrogate", 
                                  backend = surrogate_data, 
                                  target = "code")

## Measure
measure_auc <- msr("classif.auc")
measure_sen <- msr("classif.sensitivity")
measure_spe <- msr("classif.specificity")
measure_acc <- msr("classif.acc")
measure_bacc <- msr("classif.bacc")

## CV strategy
cv_rsmp <- rsmp("repeated_cv", folds = 5, repeats = 10)
cv_rsmp$instantiate(task_drugs)


lrn_rpart <- lrn("classif.rpart", predict_type = "prob")
rr_rpart <- resample(task_surrogate, lrn_rpart, cv_rsmp)

rr_rpart$score(measure_auc)
rr_rpart$aggregate(measure_auc)
rr_rpart$aggregate(measure_sen)
rr_rpart$aggregate(measure_spe)
rr_rpart$aggregate(measure_acc)
rr_rpart$aggregate(measure_bacc)


