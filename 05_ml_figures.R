## Create partial dependency and variable importance figures

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
library(ggplot2)

## ----------------------------------------------------------------------------
## Load data
load("data_for_ml.RData")

## Set all features as factors for outcomes (dialysis modality)
y_train <- final_dat %>% 
  mutate(code = as.factor(code_b),
         sex = as.numeric(sex == "F")) %>%
  select(code_b)

## Assign features to a dataframe
final_dat <- final_dat %>% 
  mutate(code = as.factor(code_b),
         sex = as.numeric(sex == "F")) %>%
  select(-patient_id, -encounter_id, -code_b, -race, -ethnicity)

## ----------------------------------------------------------------------------
##MLR3 setup
## set up task-outline of what model will do
task_drugs <- TaskClassif$new(id = "drugs", backend = final_dat, 
                              target = "code")

task_drugs$col_roles$stratum <- "code"

## ----------------------------------------------------------------------------
## Random forest model
lrn_rf_vi <- lrn("classif.ranger", predict_type = "prob",
                 mtry.ratio = 0.222222,
                 sample.fraction = 0.9,
                 num.trees = 450,
                 importance = "permutation")

lrn_rf_vi$train(task_drugs)

lrn_rf_pred <- lrn_rf_vi$predict_newdata(final_dat_test)

## ----------------------------------------------------------------------------
## Variable importance plot
vi_rf <- vip(lrn_rf_vi)$data
vi_rf$Drug <- vi_rf$Variable

nvi <- nrow(vi_rf)

## RxNorm to medication name for plot labels
drugs <- read.csv("selected_drugs.csv") 
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
print(p1)
ggsave("vi_rf.pdf", p1)

## ----------------------------------------------------------------------------
## Partial dependency plot (age)
final_dat2 <- as.data.frame(final_dat)
final_dat2$code = factor(final_dat2$code, labels = c("iHD", "CRRT"))
x <- final_dat2[which(names(final_dat2) != "code")]
model <- Predictor$new(lrn_rf_vi, data = x, y = final_dat2$code)   
eff <- FeatureEffect$new(model, feature = c("age"), method = "pdp")
p2 <- eff$plot() + theme_bw() + theme(text = element_text(size = 20))  
ggsave("pdp_rf.pdf", p2)
