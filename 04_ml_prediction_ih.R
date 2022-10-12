## Uses final trained model to predict dialysis type in PCH dataset

## Load libraries
set.seed(1234)
library(tidyverse)
library(mlr3)
library(mlr3learners)
library(mlr3viz)
library(mlr3tuning)
library(paradox)


## final_dat = TriNetX
## if_out = PCH
## Load data
load("data_for_ml.RData")

## Set all features as factors and remove columns (features) not used for model
y_train <- final_dat %>% 
  mutate(code = as.factor(code_b),
         sex = as.numeric(sex == "F")) %>%
  select(code_b)

final_dat <- final_dat %>% 
  mutate(code = as.factor(code_b),
         sex = as.numeric(sex == "F")) %>%
  select(-patient_id, -encounter_id, -code_b, -race, -ethnicity)

## Load data
load("ih_data_for_prediction.RData")

## Set all features as factors and remove columns (features) not used for model
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
## set up task-outline of what model will do
task_drugs <- TaskClassif$new(id = "drugs", backend = final_dat, 
                              target = "code")

task_drugs$col_roles$stratum <- "code"

## ----------------------------------------------------------------------------
## Helper function to try different threshold values 
cost_measure = msr("classif.bacc")
with_threshold = function(p, th) {
  p$set_threshold(th)
  list(confusion = p$confusion, costs = p$score(measures = cost_measure, 
                                                task = task_drugs))
}

## ----------------------------------------------------------------------------
## Random forest model
lrn_rf_vi <- lrn("classif.ranger", predict_type = "prob",
                 mtry.ratio = 0.222222,
                 sample.fraction = 0.9,
                 num.trees = 450)

lrn_rf_vi$train(task_drugs)

lrn_rf_pred <- lrn_rf_vi$predict_newdata(final_dat_test)

with_threshold(lrn_rf_pred, 0.5)

out.rf <- data.frame(myth = seq(0.5, 1, length.out = 100))
out.rf$costs <- rep(NA, nrow(out.rf))
for (i in 1:nrow(out.rf)) {
  out.rf$costs[i] <- with_threshold(lrn_rf_pred, out.rf$myth[i])$costs
}

plot(out.rf, type = 'l')

## ----------------------------------------------------------------------------
## xgboost model
lrn_xgb_vi <- lrn("classif.xgboost", 
                  predict_type = "prob",
                  nrounds = 750,
                  eta = exp(-6.268678),
                  max_depth = 12,
                  colsample_bytree = 0.5683879, 
                  colsample_bylevel = 0.6711491, 
                  lambda = exp(-2.583321), 
                  alpha = exp(-4.170395), 
                  #gamma = 0.5,
                  subsample = 0.8696336)
lrn_xgb_vi$train(task_drugs)

lrn_xgb_pred <- lrn_xgb_vi$predict_newdata(final_dat_test)

with_threshold(lrn_xgb_pred, 0.5)

out.xgb <- data.frame(myth = seq(0.5, 1, length.out = 100))
out.xgb$costs <- rep(NA, nrow(out.xgb))
for (i in 1:nrow(out.rf)) {
  out.xgb$costs[i] <- with_threshold(lrn_xgb_pred, out.xgb$myth[i])$costs
}

plot(out.xgb, type = 'l')

print(out.rf[which.max(out.rf$costs),])
print(out.xgb[which.max(out.xgb$costs),])

              
with_threshold(lrn_rf_pred, out.rf$myth[which.max(out.rf$costs)[1]])
with_threshold(lrn_xgb_pred, out.xgb$myth[which.max(out.xgb$costs)[1]])



