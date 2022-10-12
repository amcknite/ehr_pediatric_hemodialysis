## Load libraries
set.seed(1234)
library(data.table)
library(stringi)
library(tidyverse)

dirpath <- "/Volumes/Elements/Data/trinetx3/"

## Read in ICD code dictionary and pull out RxNorm codes
codes <- fread(paste0(dirpath, "./60771bc9c2072c76ebbe3710_20210420_135301331/standardized_terminology.csv"))
rxcodes <- codes[code_system == "RxNorm"]

## Read in medications
child_dat <- fread(paste0(dirpath, "icd10/filtered_meds.csv"))
child_dat <- child_dat[child_dat$code_system == "RxNorm", ]

# Calculate age of patients
child_dat$age <- child_dat$start_year - child_dat$year_of_birth

## ------------------------------------------------------------------------- ##
## Make table of number of the number of encounters for each medications by age
child_tab <- child_dat[, .(.N), by = .(age, code)]
child_tab$code <- as.character(child_tab$code)

child_tab <- merge(child_tab, rxcodes, by = "code", all.x = TRUE, all.y = FALSE)

child_tab <- unique(child_tab[, -c("path", "unit")])
child_tab <- child_tab[order(age, -N)]

fwrite(child_tab, file = "child_meds_by_age.csv")

## ------------------------------------------------------------------------- ##
## Make table of medication and diagnoses by age
child_tab <- child_dat[, .(.N), by = .(age, code, code_b)]
child_tab$code <- as.character(child_tab$code)

## Pivot- change table into cross table format
child_tab <- dcast(child_tab, age + code ~ code_b)
child_tab <- merge(child_tab, rxcodes, by = "code", 
                   all.x = TRUE, all.y = FALSE)
child_tab <- unique(child_tab[, -c("path", "unit")])

child_tab <- child_tab[order(age)]

fwrite(child_tab, file = "child_meds_by_age_and_proc.csv")

## ------------------------------------------------------------------------- ##
## Formating data for machine learning algorithms- ICD10

## Pull out ICD10 coded CRRT (5A1D90Z) and iHD (5A1D70Z) patient encounters
enc_proc <- as.data.frame.matrix(table(child_dat$encounter_id, child_dat$code_b))
encID <- which(enc_proc$`5A1D70Z` > 0 & enc_proc$`5A1D90Z` > 0)
enc_keep <- row.names(enc_proc)[-encID]

## Binary encode drugs (0 not given, 1 drug was given to patient)
bin_fun <- function(x) {
  if (sum(x) > 0) {
    return(1)
  } else {
    return(0)
  }
}

## Make table of race and ethnicity ICD10 coded patients
child_dat$race_ethnicity <- paste(child_dat$race, child_dat$ethnicity)
child_dat$race_ethnicity <- abbreviate(child_dat$race_ethnicity)

## Pivot- change table into cross table format
final_dat <- dcast(child_dat, encounter_id + patient_id + age + sex + race + ethnicity + code_b ~ code, 
                   fun = bin_fun, value.var = "age")

## Remove duplicate medications- results in unique medications per encounter (each administered drug only counted once per encounter)
final_dat <- final_dat %>%
  distinct(encounter_id, patient_id, .keep_all = TRUE)

## Filter out CRRT (5A1D90Z) and iHD (5A1D70Z) patient encounters
final_dat <- final_dat %>%
  filter(code_b %in% c("5A1D70Z", "5A1D90Z")) %>%
  filter(encounter_id %in% enc_keep)

## Remove all drugs given to less than 5% of patients
final_dat <- final_dat %>%
  select_if(negate(function(col) is.numeric(col) && sum(col) < nrow(final_dat) * 0.05))

## Make column names acceptable- adds RX in front of code numbers
colnames(final_dat)[8:ncol(final_dat)] <- 
  paste0("RX",colnames(final_dat)[8:ncol(final_dat)])

save(final_dat, file = "data_for_ml.RData")

stop()
