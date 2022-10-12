## Cross validation for subpopulations (SI figure)

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
  select(-encounter_id, -code_b)

all_races <- c("Black or African American", "Asian", 
               "American Indian or Alaska Native", "White", "Unknown")

all_ethnicities <- c("Hispanic or Latino", "Not Hispanic or Latino", "Unknown")

all_sexes <- c(0, 1)

## Set up task
task_drugs <- TaskClassif$new(id = "drugs", backend = final_dat2, 
                              target = "code")

# task_drugs$col_roles$stratum <- "code"
task_drugs$col_roles$group <- "patient_id"
task_drugs$set_col_roles("patient_id", remove_from = 'feature')
task_drugs$set_col_roles("race", remove_from = 'feature')
task_drugs$set_col_roles("ethnicity", remove_from = 'feature')

## Resampling
cv_rsmp <- rsmp("repeated_cv", folds = 5, repeats = 2)
cv_rsmp$instantiate(task_drugs)

## Measure
measure_auc <- msr("classif.auc")
measure_sen <- msr("classif.sensitivity")
measure_spe <- msr("classif.specificity")
measure_acc <- msr("classif.acc")
measure_bacc <- msr("classif.bacc")
all_msr <- c(measure_auc, measure_sen, measure_spe, measure_acc, measure_bacc)

## ------------------------------------------------------------------------- ##
## Random forest
rf_race <- NULL
for (race in all_races) {
  train_set = which(final_dat2$race != race)
  test_set = which(final_dat2$race == race)
  
  lrn_rf <- lrn("classif.ranger", predict_type = "prob",
                mtry.ratio = 0.222222,
                sample.fraction = 0.9,
                num.trees = 450)
  
  lrn_rf$train(task_drugs, row_ids = train_set)
  
  predict_val = lrn_rf$predict(task_drugs, row_ids = test_set)
  
  results <- predict_val$score(all_msr)
  
  rf_race <- rbind(rf_race, results)
  
}
rf_race <- data.frame(rf_race)
rownames(rf_race) <- all_races
write.csv(rf_race, "./xv_groups/rf_race.csv")

rf_eth <- NULL
for (eth in all_ethnicities) {
  train_set = which(final_dat2$ethnicity != eth)
  test_set = which(final_dat2$ethnicity == eth)
  
  lrn_rf <- lrn("classif.ranger", predict_type = "prob",
                mtry.ratio = 0.222222,
                sample.fraction = 0.9,
                num.trees = 450)
  
  lrn_rf$train(task_drugs, row_ids = train_set)
  
  predict_val = lrn_rf$predict(task_drugs, row_ids = test_set)
  
  results <- predict_val$score(all_msr)
  
  rf_eth <- rbind(rf_eth, results)
  
}
rf_eth <- data.frame(rf_eth)
rownames(rf_eth) <- all_ethnicities
write.csv(rf_eth, "./xv_groups/rf_eth.csv")

rf_sex <- NULL
for (sex in all_sexes) {
  train_set = which(final_dat2$sex != sex)
  test_set = which(final_dat2$sex == sex)
  
  lrn_rf <- lrn("classif.ranger", predict_type = "prob",
                mtry.ratio = 0.222222,
                sample.fraction = 0.9,
                num.trees = 450)
  
  lrn_rf$train(task_drugs, row_ids = train_set)
  
  predict_val = lrn_rf$predict(task_drugs, row_ids = test_set)
  
  results <- predict_val$score(all_msr)
  
  rf_sex <- rbind(rf_sex, results)
  
}
rf_sex <- data.frame(rf_sex)
rownames(rf_sex) <- all_sexes
write.csv(rf_sex, "./xv_groups/rf_sex.csv")

## ------------------------------------------------------------------------- ##
## XGBoost
xgb_race <- NULL
for (race in all_races) {
  train_set = which(final_dat2$race != race)
  test_set = which(final_dat2$race == race)
  
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
  
  lrn_xgb$train(task_drugs, row_ids = train_set)
  
  predict_val = lrn_xgb$predict(task_drugs, row_ids = test_set)
  
  results <- predict_val$score(all_msr)
  
  xgb_race <- rbind(xgb_race, results)
  
}
xgb_race <- data.frame(xgb_race)
rownames(xgb_race) <- all_races
write.csv(xgb_race, "./xv_groups/xgb_race.csv")

xgb_eth <- NULL
for (eth in all_ethnicities) {
  train_set = which(final_dat2$ethnicity != eth)
  test_set = which(final_dat2$ethnicity == eth)
  
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
  
  lrn_xgb$train(task_drugs, row_ids = train_set)
  
  predict_val = lrn_xgb$predict(task_drugs, row_ids = test_set)
  
  results <- predict_val$score(all_msr)
  
  xgb_eth <- rbind(xgb_eth, results)
  
}
xgb_eth <- data.frame(xgb_eth)
rownames(xgb_eth) <- all_ethnicities
write.csv(xgb_eth, "./xv_groups/xgb_eth.csv")

xgb_sex <- NULL
for (sex in all_sexes) {
  train_set = which(final_dat2$sex != sex)
  test_set = which(final_dat2$sex == sex)
  
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
  
  lrn_xgb$train(task_drugs, row_ids = train_set)
  
  predict_val = lrn_xgb$predict(task_drugs, row_ids = test_set)
  
  results <- predict_val$score(all_msr)
  
  xgb_sex <- rbind(xgb_sex, results)
  
}
xgb_sex <- data.frame(xgb_sex)
rownames(xgb_sex) <- all_sexes
write.csv(xgb_sex, "./xv_groups/xgb_sex.csv")

