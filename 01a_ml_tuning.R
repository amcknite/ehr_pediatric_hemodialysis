## Script to run hyperparameter tuning (cut at line 342)

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

load("data_for_ml.RData")

as.numeric(final_dat$sex == "F")
final_dat2 <- final_dat %>% 
  mutate(code = as.factor(code_b),
         sex = as.numeric(sex == "F")) %>%
  select(-patient_id, -encounter_id, -code_b, -race, -ethnicity)

## Quick RPART for viz
library(rpart)
library(rpart.plot)
fit.rpart <- rpart(code ~ ., final_dat2)
prp(fit.rpart, extra = 4)


## Set up task
task_drugs <- TaskClassif$new(id = "drugs", backend = final_dat2, 
                              target = "code")

task_drugs$col_roles$stratum <- "code"

## Measure
measure_auc <- msr("classif.auc")
measure_sen <- msr("classif.sensitivity")
measure_spe <- msr("classif.specificity")
measure_acc <- msr("classif.acc")
measure_bacc <- msr("classif.bacc")

## Resampling
set.seed(1234)
holdout_rsmp <- rsmp("holdout", ratio = 0.8)
holdout_rsmp$instantiate(task_drugs)

train_set <- holdout_rsmp$instance$train
test_set <- holdout_rsmp$instance$test

## Check stratification
prop.table(table(task_drugs$truth()))
prop.table(table(task_drugs$truth()[train_set]))

## ------------------------------------------------------------------------- ##
## Simple log regression
learner <- lrn("classif.log_reg")
learner

learner$train(task_drugs, row_ids = train_set)

## Calibration prediction
learner$predict_type <- "prob"
pred_train <- learner$predict(task_drugs, row_ids = train_set)
pred_train

pred_train$score(measure_auc)
pred_train$score(measure_sen)
pred_train$score(measure_spe)

autoplot(pred_train, type = 'roc')

## Calibration prediction
learner$predict_type <- "prob"
pred_test <- learner$predict(task_drugs, row_ids = test_set)
pred_test

pred_test$score(measure_auc)
pred_test$score(measure_sen)
pred_test$score(measure_spe)

autoplot(pred_test, type = 'roc')

## Resampling
cv_rsmp <- rsmp("repeated_cv", folds = 5, repeats = 3)
cv_rsmp$instantiate(task_drugs)

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

lrn_rf$train(task_drugs, row_ids = train_set)

# x = final_dat2[which(names(final_dat2) != "code")]
# model = Predictor$new(lrn_rf, data = x[train_set, ], y = final_dat2$code[train_set])
# 
# effect = FeatureEffects$new(model)
# plot(effect, features = c("bill_length"))

## Calibration prediction
lrn_rf$predict_type <- "prob"
pred_train <- lrn_rf$predict(task_drugs, row_ids = train_set)
pred_train

measure_auc <- msr("classif.auc")
pred_train$score(measure_auc)

autoplot(pred_train, type = 'roc')

## Calibration prediction
lrn_rf$predict_type <- "prob"
pred_test <- lrn_rf$predict(task_drugs, row_ids = test_set)
pred_test

measure_auc <- msr("classif.auc")
pred_test$score(measure_auc)

autoplot(pred_test, type = 'roc')

rr <- resample(task_drugs, lrn_rf, cv_rsmp, store_models = TRUE)
rr$score(measure_auc)
rr$aggregate(measure_auc)
rr$aggregate(measure_sen)
rr$aggregate(measure_spe)

## Tuning 
## Strategy
resampling_inner = rsmp("holdout", ratio = 0.8)
resampling_outer = rsmp("cv", folds = 3)

## Parameter set
tune_ps = ParamSet$new(list(
  ParamInt$new("mtry", lower = 10, upper = 20),
  ParamInt$new("num.trees", lower = 200, upper = 2000)
))

## Evaluation limit
evals = trm("evals", n_evals = 100)

## Tuner
tuner = tnr("grid_search", resolution = 11)

## Set autotuner
at_rf = AutoTuner$new(learner = lrn_rf, 
                      resampling = resampling_inner,
                      measure = measure_auc, 
                      search_space = tune_ps,
                      terminator = evals,
                      tuner = tuner,
                      store_models = TRUE)

## Run outer loop
# future::plan("multisession")
# rr_rf = resample(task = task_drugs, learner = at_rf,
#                  resampling = resampling_outer, store_models = TRUE)

# rr_rf$score(measure_auc)
# rr_rf$aggregate(measure_auc)
# 
# rr_rf$learners[[1]]$param_set$values

# lrn_rf_vi <- lrn("classif.ranger", predict_type = "prob", 
#                  mtry = rr_rf$learners[[1]]$param_set$values$mtry, 
#                  num.trees = rr_rf$learners[[1]]$param_set$values$num.trees)
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
# final_dat2$code_b <- ifelse(final_dat2$code_b == "5A1D90Z", 1, 0)
# final_dat2$sex <- ifelse(final_dat2$sex == "F", 1, 0)
## Set up task
# task_drugs <- TaskClassif$new(id = "drugs", backend = final_dat2, 
#                               target = "code")
# task_drugs$col_roles$stratum <- "code"


lrn_xgb <- lrn("classif.xgboost", predict_type = "prob")
rr <- resample(task_drugs, lrn_xgb, cv_rsmp, store_models = TRUE)
rr$score(measure_auc)
rr$aggregate(measure_auc)

## Tuning 
## Strategy
# resampling_inner = rsmp("holdout", ratio = 0.8)
# resampling_outer = rsmp("cv", folds = 3)
resampling_inner = rsmp("cv", folds = 3)
resampling_outer = rsmp("cv", folds = 5)

## Parameter set
tune_ps = ParamSet$new(list(
  ParamDbl$new("eta", lower = 0.001, upper = 0.1),
  ParamInt$new("max_depth", lower = 1, upper = 10), ## Raise upper limit
  ParamInt$new("nrounds", lower = 100, upper = 1000)
))

## Evaluation limit
evals = trm("evals", n_evals = 50)

## Tuner
tuner = tnr("grid_search", resolution = 10)

## Set autotuner
at_xgb = AutoTuner$new(learner = lrn_xgb, 
                       resampling = resampling_inner,
                       measure = measure_auc, 
                       search_space = tune_ps,
                       terminator = evals,
                       tuner = tuner)

## Run outer loop
# future::plan(list("multisession", "sequential"))
# rr_xgb = resample(task = task_drugs, learner = at_xgb,
#                  resampling = resampling_outer, store_models = TRUE)

# rr_xgb$score(measure_auc)
# rr_xgb$aggregate(measure_auc)
# 
# rr_xgb$learners[[1]]$param_set$values
# 

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

stop()

## Dump out VIP scores
library(data.table)
dirpath <- "/Volumes/Elements/Data/trinetx3/"
codes <- fread(paste0(dirpath, "./60771bc9c2072c76ebbe3710_20210420_135301331/standardized_terminology.csv"))
rxcodes <- codes[code_system == "RxNorm"]

vip_xgb <- vip(lrn_xgb_vi$model, num_features = 11)
vip_xgb$data$drug <- rep(NA, 11)
for (i in 1:nrow(vip_xgb$data)) {
  rxID <- which(rxcodes$code == substr(vip_xgb$data$Variable[i], 3, 12))
  if (length(rxID) > 0) {
    vip_xgb$data$drug[i] <- rxcodes$code_description[rxID[1]]
  } else {
    vip_xgb$data$drug[i] <- vip_xgb$data$Variable[i]
  }
}
write.csv(vip_xgb$data, "vip_xgb.csv", row.names = FALSE)

library(iml)
final_dat2 <- as.data.frame(final_dat2)
x <- final_dat2[which(names(final_dat2) != "code")]
model <- Predictor$new(lrn_rf_vi, data = x, y = final_dat2$code)   

# pdf("pdp_age_xgb.pdf")

## Partial dependency plots
png("pdp_age_rf.png", width = 800, height = 600)
eff <- FeatureEffect$new(model, feature = c("age"), method = "pdp")
eff$plot() + theme_bw()
dev.off()

pdf("pdp_age_rf.pdf")
eff <- FeatureEffect$new(model, feature = c("age"), method = "pdp")
eff$plot() + theme_bw()
dev.off()

stop()

library(iml)
final_dat2 <- as.data.frame(final_dat2)
x <- final_dat2[which(names(final_dat2) != "code")]
model <- Predictor$new(lrn_xgb_vi, data = x, y = final_dat2$code)   

# pdf("pdp_age_xgb.pdf")

## Partial dependency plots
png("pdp_age_xgb.png", width = 800, height = 600)
eff <- FeatureEffect$new(model, feature = c("age"), method = "pdp")
eff$plot() + theme_bw()
dev.off()

pdf("pdp_drugs_xgb.pdf")

eff <- FeatureEffect$new(model, feature = c("RX1721684"), method = "pdp")
eff$plot() + theme_bw()

eff <- FeatureEffect$new(model, feature = c("RX727373"), method = "pdp")
eff$plot() + theme_bw()

eff <- FeatureEffect$new(model, feature = c("RX1807630"), method = "pdp")
eff$plot() + theme_bw()

eff <- FeatureEffect$new(model, feature = c("RX828527"), method = "pdp")
eff$plot() + theme_bw()

eff <- FeatureEffect$new(model, feature = c("RX730781"), method = "pdp")
eff$plot() + theme_bw()

eff <- FeatureEffect$new(model, feature = c("RX238082"), method = "pdp")
eff$plot() + theme_bw()

eff <- FeatureEffect$new(model, feature = c("RX847630"), method = "pdp")
eff$plot() + theme_bw()

eff <- FeatureEffect$new(model, feature = c("RX313932"), method = "pdp")
eff$plot() + theme_bw()

eff <- FeatureEffect$new(model, feature = c("RX308135"), method = "pdp")
eff$plot() + theme_bw()

eff <- FeatureEffect$new(model, feature = c("RX261283"), method = "pdp")
eff$plot() + theme_bw()

dev.off()


effect = FeatureEffects$new(model)
plot(effect, features = c("age")) + theme_bw()

x <- final_dat2[, -379]
y <- as.numeric(final_dat2$code == "5A1D90Z")
y <- final_dat2$code == "5A1D90Z"
model <- Predictor$new(lrn_xgb_vi$model, data = x, y = y)   

## Doesn't work with xgb
model <- Predictor$new(lrn_xgb_vi, data = x, y = y)   
dt <- TreeSurrogate$new(model)

lrn_xgb_pred <- lrn_xgb_vi$predict(task_drugs)

library(rpart)
library(rpart.plot)
x <- final_dat2[, -231]
surrogate_data <- cbind(x, lrn_xgb_pred$data$response)
names(surrogate_data)[ncol(surrogate_data)] <- "code"
lrn_rpart <- rpart(code ~., surrogate_data, 
                   control = rpart.control(maxdepth = 3))
lrn_rpart
prp(lrn_rpart)
stop()

lrn_rpart2 <- rpart(code ~., surrogate_data, 
                   control = rpart.control(maxdepth = 10))
lrn_rpart2
prp(lrn_rpart2)

## Surrogate tree
library(flashlight)
# Data wrapper for xgboost
prep_xgb <- function(data, x) {
  data %>%
    select_at(x) %>%
    mutate_if(Negate(is.numeric), as.integer) %>%
    data.matrix()
}
x <- names(final_dat2)[-379]
y <- as.numeric(final_dat2$code == "5A1D90Z")
dtrain <- xgboost::xgb.DMatrix(prep_xgb(final_dat2, x), 
                               label = y)
# dvalid <- xgboost::xgb.DMatrix(prep_xgb(valid, x), 
#                                label = valid[["log_price"]])

params <- list(eta = 0.067,
               max_depth = 1)

fit_xgb <- xgboost::xgb.train(
  params,
  data = dtrain,
  # watchlist = list(train = dtrain, valid = dvalid),
  nrounds = 900,
  print_every_n = 25,
  objective = "binary:logistic"
)

fl_xgb <- flashlight(
  model = fit_xgb, 
  label = "xgb",
  predict_function = function(mod, X) predict(mod, prep_xgb(X, x))
)
print(fl_xgb)
surr <- light_global_surrogate(fl_xgb, v = x)

