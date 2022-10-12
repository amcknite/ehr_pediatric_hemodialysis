## Runs cross-validation (blocked on patient id)

set.seed(42)
library(tidyverse)
library(mlr3)
library(mlr3learners)
# library(mlr3measures)
library(mlr3pipelines)
library(mlr3viz)
library(mlr3tuning)
library(paradox)
library(ranger)
library(vip)
library(pdp)

load("data_for_ml.RData")

as.numeric(final_dat$sex == "F")
final_dat2 <- final_dat %>% 
  mutate(code = as.factor(code_b),
         sex = as.numeric(sex == "F")) %>%
  select(-encounter_id, -code_b, -race, -ethnicity)

## Set up task
task_drugs <- TaskClassif$new(id = "drugs", backend = final_dat2, 
                              target = "code")

# task_drugs$col_roles$stratum <- "code"
task_drugs$col_roles$group <- "patient_id"
task_drugs$set_col_roles("patient_id", remove_from = 'feature')

## Resampling
cv_rsmp <- rsmp("repeated_cv", folds = 5, repeats = 10)
cv_rsmp$instantiate(task_drugs)

## Measure
measure_auc <- msr("classif.auc")
measure_sen <- msr("classif.sensitivity")
measure_spe <- msr("classif.specificity")
measure_acc <- msr("classif.acc")
measure_bacc <- msr("classif.bacc")

## ------------------------------------------------------------------------- ##
## Featureless
learner <- lrn("classif.featureless", predict_type = "prob")
rr_fl <- resample(task_drugs, learner, cv_rsmp)
rr_fl$score(measure_auc)
rr_fl$aggregate(measure_auc)
rr_fl$aggregate(measure_sen)
rr_fl$aggregate(measure_spe)
rr_fl$aggregate(measure_acc)
rr_fl$aggregate(measure_bacc)

## ------------------------------------------------------------------------- ##
## Simple log regression
learner <- lrn("classif.log_reg", predict_type = "prob")
rr_lr <- resample(task_drugs, learner, cv_rsmp)
rr_lr$score(measure_auc)
rr_lr$aggregate(measure_auc)
rr_lr$aggregate(measure_sen)
rr_lr$aggregate(measure_spe)
rr_lr$aggregate(measure_acc)
rr_lr$aggregate(measure_bacc)

## ------------------------------------------------------------------------- ##
## Naive Bayes
learner <- lrn("classif.naive_bayes", predict_type = "prob")
rr_nb <- resample(task_drugs, learner, cv_rsmp)
rr_nb$score(measure_auc)
rr_nb$aggregate(measure_auc)
rr_nb$aggregate(measure_sen)
rr_nb$aggregate(measure_spe)
rr_nb$aggregate(measure_acc)
rr_nb$aggregate(measure_bacc)

## ------------------------------------------------------------------------- ##
## GLMnet
learner <- lrn("classif.glmnet", predict_type = "prob",
               alpha = 0.222222, s = exp(-5.116856))
rr_glmnet <- resample(task_drugs, learner, cv_rsmp)
rr_glmnet$score(measure_auc)
rr_glmnet$aggregate(measure_auc)
rr_glmnet$aggregate(measure_sen)
rr_glmnet$aggregate(measure_spe)
rr_glmnet$aggregate(measure_acc)
rr_glmnet$aggregate(measure_bacc)

## ------------------------------------------------------------------------- ##
## k-NN
learner <- lrn("classif.kknn", predict_type = "prob")
rr_knn <- resample(task_drugs, learner, cv_rsmp)
rr_knn$score(measure_auc)
rr_knn$aggregate(measure_auc)
rr_knn$aggregate(measure_sen)
rr_knn$aggregate(measure_spe)
rr_knn$aggregate(measure_acc)
rr_knn$aggregate(measure_bacc)

## ------------------------------------------------------------------------- ##
## SVM
learner <- lrn("classif.svm", kernel = "radial", scale = TRUE, 
               predict_type = "prob", gamma = 0.01)
rr_svm <- resample(task_drugs, learner, cv_rsmp)
rr_svm$score(measure_auc)
rr_svm$aggregate(measure_auc)
rr_svm$aggregate(measure_sen)
rr_svm$aggregate(measure_spe)
rr_svm$aggregate(measure_acc)
rr_svm$aggregate(measure_bacc)

## ------------------------------------------------------------------------- ##
## Random forest
lrn_rf <- lrn("classif.ranger", predict_type = "prob",
              mtry.ratio = 0.222222,
              sample.fraction = 0.9,
              num.trees = 450)
rr_rf <- resample(task_drugs, lrn_rf, cv_rsmp, store_models = TRUE)
rr_rf$score(measure_auc)
rr_rf$aggregate(measure_auc)
rr_rf$aggregate(measure_sen)
rr_rf$aggregate(measure_spe)
rr_rf$aggregate(measure_acc)
rr_rf$aggregate(measure_bacc)

## ------------------------------------------------------------------------- ##
## XGBoost
lrn_xgb <- lrn("classif.xgboost", 
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
rr_xgb <- resample(task_drugs, lrn_xgb, cv_rsmp, store_models = TRUE)
rr_xgb$score(measure_auc)
rr_xgb$aggregate(measure_auc)
rr_xgb$aggregate(measure_sen)
rr_xgb$aggregate(measure_spe)
rr_xgb$aggregate(measure_acc)
rr_xgb$aggregate(measure_bacc)

## ------------------------------------------------------------------------- ##
## Table
cols <- c("classif.auc", 
          "classif.sensitivity",
          "classif.specificity",
          "classif.acc",
          "classif.bacc")
all_msr <- c(measure_auc, measure_sen, measure_spe, measure_acc, measure_bacc)

all_rr <- rbind(rr_fl$score(all_msr),
                rr_lr$score(all_msr),
                rr_nb$score(all_msr),
                rr_glmnet$score(all_msr),
                rr_knn$score(all_msr),
                rr_svm$score(all_msr),
                rr_rf$score(all_msr),
                rr_xgb$score(all_msr)
                )

result_mean <- all_rr[, sapply(.SD, function(x) list(mean = mean(x))), 
              .SDcols = cols, by = learner_id]

result_sd <- all_rr[, sapply(.SD, function(x) list(sd = sd(x))), 
                      .SDcols = cols, by = learner_id]


result_all <- data.frame(learner_id = result_mean$learner_id,
                         result_mean[, 2], result_sd[, 2],
                         result_mean[, 3], result_sd[, 3],
                         result_mean[, 4], result_sd[, 4],
                         result_mean[, 5], result_sd[, 5],
                         result_mean[, 6], result_sd[, 6]
)

knitr::kable(result_all)
write.csv(result_all, "model_xvalidation.csv",
          row.names = FALSE)

