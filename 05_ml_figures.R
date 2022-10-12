## Creates partial dependency and variable importance figures

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
library(ggplot2)

load("data_for_ml.RData")

y_train <- final_dat %>% 
  mutate(code = as.factor(code_b),
         sex = as.numeric(sex == "F")) %>%
  select(code_b)

final_dat <- final_dat %>% 
  mutate(code = as.factor(code_b),
         sex = as.numeric(sex == "F")) %>%
  select(-patient_id, -encounter_id, -code_b, -race, -ethnicity)

## ----------------------------------------------------------------------------
##MLR3 setup
## Set up task
task_drugs <- TaskClassif$new(id = "drugs", backend = final_dat, 
                              target = "code")

task_drugs$col_roles$stratum <- "code"

## ----------------------------------------------------------------------------
## Ranger model
lrn_rf_vi <- lrn("classif.ranger", predict_type = "prob",
                 mtry.ratio = 0.222222,
                 sample.fraction = 0.9,
                 num.trees = 450,
                 importance = "permutation")

lrn_rf_vi$train(task_drugs)

lrn_rf_pred <- lrn_rf_vi$predict_newdata(final_dat_test)

## Make plot
drugs <- read.csv("selected_drugs.csv")
vi_rf <- vip(lrn_rf_vi)$data
vi_rf$Drug <- vi_rf$Variable

nvi <- nrow(vi_rf)

for (i in 1:nvi) {
  if (substr(vi_rf$Variable[i], 0, 2) == "RX") {
    rxnn <- substr(vi_rf$Variable[i], 3, 20)
    drug_id <- which(drugs$rxcode == rxnn)
    vi_rf$Drug[i] <- drugs$name[drug_id]
  }
}

p1 <- ggplot(vi_rf, aes(x = Importance, y = reorder(Drug, Importance, mean))) +
  geom_col() +
  scale_y_discrete("Drug") +
  theme_minimal() +
  theme(text = element_text(size = 20))
pdf("test.pdf")
print(p1)
dev.off()
ggsave("vi_rf.pdf", p1)

## PDP (age)
final_dat2 <- as.data.frame(final_dat)
final_dat2$code = factor(final_dat2$code, labels = c("iHD", "CRRT"))
x <- final_dat2[which(names(final_dat2) != "code")]

task_drugs <- TaskClassif$new(id = "drugs", backend = final_dat2, 
                              target = "code")
lrn_rf_vi <- lrn("classif.ranger", predict_type = "prob",
                 mtry.ratio = 0.222222,
                 sample.fraction = 0.9,
                 num.trees = 450,
                 importance = "permutation")
lrn_rf_vi$train(task_drugs)

model <- Predictor$new(lrn_rf_vi, data = x, y = final_dat2$code)   

eff <- FeatureEffect$new(model, feature = c("age"), method = "pdp")
p2 <- eff$plot() + theme_bw() + theme(text = element_text(size = 20))  

ggsave("pdp_rf.pdf", p2)

stop()
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

## RF predictions

