## Script to tune threshold to discriminate iHD and CRRT

## Load libraries
set.seed(1234)
library(data.table)
library(tidyverse)
library(mlr3verse)
library(mlr3tuning)
library(mlr3tuningspaces)
library(paradox)
library(ranger)
library(vip)
library(pdp)

## -------------------------------------------------------------------------- ##
## Load pre-processed data
load("data_for_ml.RData")

## Set all features as factors and remove columns (features) not used for model
as.numeric(final_dat$sex == "F")
final_dat2 <- final_dat %>% 
  mutate(code = as.factor(code_b),
         sex = as.numeric(sex == "F")) %>%
  select(-encounter_id, -code_b, -race, -ethnicity)

## -------------------------------------------------------------------------- ##
## MLR3

## Set up task-outline of what algorithms will do
task_drugs <- TaskClassif$new(id = "drugs", backend = final_dat2, 
                              target = "code")
task_drugs$col_roles$group <- "patient_id"
task_drugs$set_col_roles("patient_id", remove_from = 'feature')

# print keys and learners
as.data.table(mlr_tuning_spaces)

## -------------------------------------------------------------------------- ##
## Random forest algorithm tuning
# tune learner with default search space
instance_rf = tune(
  method = "grid_search",
  task = task_drugs,
  learner = lts(lrn("classif.ranger")),
  resampling = rsmp ("cv", folds = 5),
  measure = msr("classif.bacc"),
  term_evals = 100
)

# best performing hyperparameter configuration
instance_rf$result

## -------------------------------------------------------------------------- ##
## Xgboost algorithm tuning
# tune learner with default search space
instance_xgb = tune(
  method = "random_search",
  task = task_drugs,
  learner = lts(lrn("classif.xgboost")),
  resampling = rsmp ("cv", folds = 5),
  measure = msr("classif.bacc"),
  term_evals = 200
)

# best performing hyperparameter configuration
instance_xgb$result

save(instance_glmnet, instance_rf, instance_xgb, 
     file = "tuned_models.RData")
