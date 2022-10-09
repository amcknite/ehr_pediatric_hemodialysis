set.seed(1234)
library(data.table)
library(stringi)
library(tidyverse)

dirpath <- "/Volumes/Elements/Data/trinetx3/"

## Med codes
codes <- fread(paste0(dirpath, "./60771bc9c2072c76ebbe3710_20210420_135301331/standardized_terminology.csv"))
rxcodes <- codes[code_system == "RxNorm"]

## Read in data
child_dat <- fread(paste0(dirpath, "icd09/filtered_meds.csv"))

child_dat$age <- child_dat$start_year - child_dat$year_of_birth

## Last ten years
# child_dat <- child_dat %>% 
#   filter(start_year >= 2010)

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

## Pivot
child_tab <- dcast(child_tab, age + code ~ code_b)
child_tab <- merge(child_tab, rxcodes, by = "code", 
                   all.x = TRUE, all.y = FALSE)
child_tab <- unique(child_tab[, -c("path", "unit")])

# child_tab <- unique(child_tab[, -c("path", "unit")])
child_tab <- child_tab[order(age)]

fwrite(child_tab, file = "child_meds_by_age_and_proc.csv")

## ------------------------------------------------------------------------- ##
## Table for ML

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

## Remove all drugs given to less than ?10% of patients
final_dat <- final_dat %>%
  select_if(negate(function(col) is.numeric(col) && sum(col) < 25))

## Make column names acceptable
colnames(final_dat)[8:ncol(final_dat)] <- 
  paste0("RX",colnames(final_dat)[8:ncol(final_dat)])

## Rename to avoid conflicts
final_dat_pred <- final_dat

save(final_dat_pred, file = "data_for_ml_predict.RData")
