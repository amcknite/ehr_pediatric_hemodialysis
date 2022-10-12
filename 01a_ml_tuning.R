## Script to run hyperparameter tuning (superseded by script 01b)

set.seed(1234)
library(tidyverse)
library(mlr3)
library(mlr3learners)
# library(mlr3measures)
library(mlr3viz)
library(mlr3tuning)
library(paradox)
library(ranger)
library(vip)
library(pdp)

## -------------------------------------------------------------
## Load data
load("data_for_ml.RData")

## Encode factors and remove extra variables
as.numeric(final_dat$sex == "F")
final_dat2 <- final_dat %>% 
  mutate(code = as.factor(code_b),
         sex = as.numeric(sex == "F")) %>%
  select(-patient_id, -encounter_id, -code_b, -race, -ethnicity)

## -------------------------------------------------------------
## ML set up 

## 1. define task
task_drugs <- TaskClassif$new(id = "drugs", backend = final_dat2, 
                              target = "code")

task_drugs$col_roles$stratum <- "code"

## 2. Define performance measure
measure_auc <- msr("classif.auc")
measure_sen <- msr("classif.sensitivity")
measure_spe <- msr("classif.specificity")
measure_acc <- msr("classif.acc")
measure_bacc <- msr("classif.bacc")

## 3. Define resampling 
cv_rsmp <- rsmp("repeated_cv", folds = 5, repeats = 10)
cv_rsmp$instantiate(task_drugs)

## ------------------------------------------------------------------------- ##
## Algorithms

## Simple log regression
learner <- lrn("classif.log_reg")
rr <- resample(task_drugs, learner, cv_rsmp, store_models = TRUE)
rr$score(measure_auc)
rr$aggregate(measure_auc)
rr$aggregate(measure_sen)
rr$aggregate(measure_spe)
rr$aggregate(measure_acc)
rr$aggregate(measure_bacc)

## ------------------------------------------------------------------------- ##
## Featureless
learner <- lrn("classif.featureless", predict_type = "prob")
rr <- resample(task_drugs, learner, cv_rsmp)
rr$score(measure_auc)
rr$aggregate(measure_auc)
rr$aggregate(measure_sen)
rr$aggregate(measure_spe)
rr$aggregate(measure_acc)
rr$aggregate(measure_bacc)

## ------------------------------------------------------------------------- ##
## Naive Bayes
learner <- lrn("classif.naive_bayes", predict_type = "prob")
rr <- resample(task_drugs, learner, cv_rsmp)
rr$score(measure_auc)
rr$aggregate(measure_auc)
rr$aggregate(measure_sen)
rr$aggregate(measure_spe)
rr$aggregate(measure_acc)
rr$aggregate(measure_bacc)

## ------------------------------------------------------------------------- ##
## GLMnet
learner <- lrn("classif.cv_glmnet", predict_type = "prob")
rr <- resample(task_drugs, learner, cv_rsmp)
rr$score(measure_auc)
rr$aggregate(measure_auc)
rr$aggregate(measure_sen)
rr$aggregate(measure_spe)
rr$aggregate(measure_acc)
rr$aggregate(measure_bacc)

## ------------------------------------------------------------------------- ##
## k-NN
learner <- lrn("classif.kknn", predict_type = "prob")
rr <- resample(task_drugs, learner, cv_rsmp)
rr$score(measure_auc)
rr$aggregate(measure_auc)
rr$aggregate(measure_sen)
rr$aggregate(measure_spe)
rr$aggregate(measure_acc)
rr$aggregate(measure_bacc)

## ------------------------------------------------------------------------- ##
## SVM
## Set up task
learner <- lrn("classif.svm", kernel = "radial", scale = TRUE, 
               predict_type = "prob", gamma = 0.01)
rr <- resample(task_drugs, learner, cv_rsmp)
rr$score(measure_auc)
rr$aggregate(measure_auc)
rr$aggregate(measure_sen)
rr$aggregate(measure_spe)
rr$aggregate(measure_acc)
rr$aggregate(measure_bacc)

## ------------------------------------------------------------------------- ##
## Random forest

lrn_rf <- lrn("classif.ranger", predict_type = "prob", importance = "permutation")
lrn_rf_vi <- lrn("classif.ranger", predict_type = "prob",
                 mtry = 16,
                 num.trees = 200)
rr <- resample(task_drugs, lrn_rf_vi, cv_rsmp, store_models = TRUE)
rr$score(measure_auc)
rr$aggregate(measure_auc)
rr$aggregate(measure_sen)
rr$aggregate(measure_spe)
rr$aggregate(measure_acc)
rr$aggregate(measure_bacc)

lrn_rf_vi <- lrn("classif.ranger", 
                 predict_type = "prob", 
                 mtry = 12, 
                 num.trees = 500, importance = "permutation")
lrn_rf_vi$train(task_drugs)
# pdf("vip_rf.pdf")
png("vip_rf.png", width = 800, height = 600)
vip(lrn_rf_vi$model, num_features = 11) + theme_bw()
dev.off()

## ------------------------------------------------------------------------- ##
## xgboost
lrn_xgb <- lrn("classif.xgboost", 
               predict_type = "prob",
               gamma = 0.75, 
               # alpha = 1, 
               # lambda = 1, 
               eta = 0.01,
               nrounds = 1200,
               subsample = 0.5,
               colsample_bytree = 0.25, 
               min_child_weight = 2,
               max_depth = 2)
rr <- resample(task_drugs, lrn_xgb, cv_rsmp, store_models = TRUE)
rr$score(measure_auc)
rr$aggregate(measure_auc)
rr$aggregate(measure_sen)
rr$aggregate(measure_spe)
rr$aggregate(measure_acc)
rr$aggregate(measure_bacc)

lrn_xgb_vi <- lrn("classif.xgboost", 
                  predict_type = "prob",
                  gamma = 0.75, 
                  # alpha = 1, 
                  # lambda = 1, 
                  eta = 0.01,
                  nrounds = 1200,
                  subsample = 0.5,
                  colsample_bytree = 0.25, 
                  min_child_weight = 2,
                  max_depth = 2)
lrn_xgb_vi$train(task_drugs)
#pdf("vip_xgb.pdf")
png("vip_xgb.png", width = 800, height = 600)
vip(lrn_xgb_vi$model, num_features = 11)  + theme_bw()
dev.off()

