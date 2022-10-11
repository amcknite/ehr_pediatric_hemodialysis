set.seed(1234)
library(data.table)
library(stringi)
library(tidyverse)

dirpath <- "/Volumes/Elements/Data/trinetx3/"

## Read in ICD code dictionary and pull out RxNorm codes
#meds<- fread(paste0(dirpath, "./60771bc9c2072c76ebbe3710_20210420_135301331/medication_drug.csv"))
#mi_at<- fread(paste0(dirpath, "./60771bc9c2072c76ebbe3710_20210420_135301331/medication_ingredient.csv"))
codes <- fread(paste0(dirpath, "./60771bc9c2072c76ebbe3710_20210420_135301331/standardized_terminology.csv"))
rxcodes <- codes[code_system == "RxNorm"]

## Read in data
child_dat <- fread(paste0(dirpath, "icd10/filtered_meds.csv"))
child_dat <- child_dat[child_dat$code_system == "RxNorm", ]

child_dat$age <- child_dat$start_year - child_dat$year_of_birth

## ------------------------------------------------------------------------- ##
## First make table for all drugs/codes
child_tab <- child_dat[, .(.N), by = .(age, code)]
child_tab$code <- as.character(child_tab$code)

child_tab <- merge(child_tab, rxcodes, by = "code", all.x = TRUE, all.y = FALSE)

child_tab <- unique(child_tab[, -c("path", "unit")])
child_tab <- child_tab[order(age, -N)]

fwrite(child_tab, file = "child_meds_by_age.csv")

## ------------------------------------------------------------------------- ##
## Now make table by code 
child_tab <- child_dat[, .(.N), by = .(age, code, code_b)]
child_tab$code <- as.character(child_tab$code)

## Pivot- switch rows to columns
child_tab <- dcast(child_tab, age + code ~ code_b)
child_tab <- merge(child_tab, rxcodes, by = "code", 
                   all.x = TRUE, all.y = FALSE)
child_tab <- unique(child_tab[, -c("path", "unit")])

# child_tab <- unique(child_tab[, -c("path", "unit")])
child_tab <- child_tab[order(age)]

fwrite(child_tab, file = "child_meds_by_age_and_proc.csv")

## ------------------------------------------------------------------------- ##
## Table for ML

## First get list of codes by patient
enc_proc <- as.data.frame.matrix(table(child_dat$encounter_id, child_dat$code_b))
encID <- which(enc_proc$`5A1D70Z` > 0 & enc_proc$`5A1D90Z` > 0)
enc_keep <- row.names(enc_proc)[-encID]

## Now make up wide table with drugs by encounter
bin_fun <- function(x) {
  if (sum(x) > 0) {
    return(1)
  } else {
    return(0)
  }
}

child_dat$race_ethnicity <- paste(child_dat$race, child_dat$ethnicity)
child_dat$race_ethnicity <- abbreviate(child_dat$race_ethnicity)
# x1 <- dcast(child_dat, encounter_id + patient_id + age + sex + code_b ~ code, value.var = "age")
## Pivot table
final_dat <- dcast(child_dat, encounter_id + patient_id + age + sex + race + ethnicity + code_b ~ code, 
                   fun = bin_fun, value.var = "age")
## Remove duplicates (age)
final_dat <- final_dat %>%
  distinct(encounter_id, patient_id, .keep_all = TRUE)

## Select only the two codes we're interested in
final_dat <- final_dat %>%
  filter(code_b %in% c("5A1D70Z", "5A1D90Z")) %>%
  filter(encounter_id %in% enc_keep)

## Remove all drugs given to less than 5% of patients
# final_dat <- final_dat %>%
#   select_if(negate(function(col) is.numeric(col) && sum(col) < 25))

final_dat <- final_dat %>%
  select_if(negate(function(col) is.numeric(col) && sum(col) < nrow(final_dat) * 0.05))

## Make column names acceptable
colnames(final_dat)[8:ncol(final_dat)] <- 
  paste0("RX",colnames(final_dat)[8:ncol(final_dat)])

save(final_dat, file = "data_for_ml.RData")

stop()
## Write out list of drugs + names
final_drugs <- names(final_dat)[8:ncol(final_dat)]
final_names <- rep(NA, length(final_drugs))

for (i in 1:length(final_drugs)) {
  
  tmp_code <- substr(final_drugs[i], 3, 20)
  
  tmp_id <- which(rxcodes$code == tmp_code)

}


stop()
library(randomForest)

final_dat$code_b <- as.factor(final_dat$code_b)
fit.rf <- randomForest(code_b ~ ., final_dat[, -c(1,2)])

library(vip)
p1 <- vip(fit.rf)
ggsave("rf_vip.pdf", p1)
library(pdp)
partial(fit.rf, "age", plot.engine = "ggplot2", 
        plot = TRUE, probs = TRUE)

partial(fit.rf, "RX1901", plot.engine = "ggplot2", 
        plot = TRUE, probs = TRUE)

stop()

library(DALEX)
library(DALEXtra)

final_dat$code_b <- as.numeric(as.factor(final_dat$code_b)) - 1

explain_rf <- explain(fit.rf,
                      data = final_dat,
                      y = final_dat$code_b,
                      label = "RF Classification",
                      colorize = FALSE)

vi <- variable_importance(explain_rf)
plot(vi)

ve_p <- variable_profile(explain_rf, variables = "age", type = "partial")
ve_p$color = "_label_"
plot(ve_p)

ve_p <- variable_profile(explain_rf, variables = "RX26225", type = "partial")
ve_p$color = "_label_"
plot(ve_p)


my_enc_id <- 10
bd <- variable_attribution(explain_rf, final_dat[my_enc_id,], type = "break_down")
plot(bd)


## Classification tree for giggles
library(rpart)
library(rpart.plot)
final_dat2 <- final_dat %>%
  select(-encounter_id, -patient_id)

fit.rpart <- rpart(code_b ~ ., final_dat2)
prp(fit.rpart, extra = 103)

pdf("trinetx_prp.pdf")
prp(fit.rpart, extra = 2)
dev.off()

