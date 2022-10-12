## Script to prepare data for ML model testing- PCH dataset

## Load libraries
set.seed(1234)
library(data.table)
library(stringi)
library(tidyverse)

## Read in data for PCH dataset
load("data_for_ml.RData")
#old_ih_patients <- read.csv("../../../ih_data/ID_patient_encounter_test.csv")
ih_patients <- read.csv("../../../ih_data/CRRT_iHD_identified_ICD10_ICD9_PCH_20210711.csv")
ih_demographics <- read.csv("../../../ih_data/IH_CRRT_PD_Demographics_20200204.csv")
ih_meds <- read.csv("../../../ih_data/IH_CRRT_PD_Medications_20200204.csv")
drug_tab <- read.csv("../../../ih_data/selected_drugs_AMM.csv")

## Pull out patient encounters coded for ICD10 CRRT(5A1D90Z) and iHD(5A1D70Z)
ih_patients <- ih_patients %>% 
  select(PATIENT_ID, ENCNTR_SEQ, ICD_5A1D70Z_FLG, ICD_5A1D90Z_FLG) %>%
  filter(ICD_5A1D70Z_FLG == 1 | ICD_5A1D90Z_FLG == 1) %>%
  mutate(sum = ICD_5A1D70Z_FLG + ICD_5A1D90Z_FLG) %>%
  filter(sum < 2) %>%
  mutate(Flag = ifelse(ICD_5A1D70Z_FLG == 1, "5A1D70Z", "5A1D90Z")) %>%
  select(-sum , -ICD_5A1D70Z_FLG, -ICD_5A1D90Z_FLG)

## Add in patient demographics (age and sex)
ih_dat <- merge(ih_patients, ih_demographics, by = c("PATIENT_ID", "ENCNTR_SEQ"))
ih_dat <- ih_dat %>%
  select(PATIENT_ID, ENCNTR_SEQ, Flag, AGE_YRS, SEX_DSP)

# Make table of medications (code) and diagnoses (code_b) by age
nenctr = nrow(ih_dat)
drug_list <- names(final_dat)[8:ncol(final_dat)]

## Create list of drug codes 
drug_tab <- pivot_longer(drug_tab, cols = name:X.24, values_drop_na = TRUE) %>%
  filter(value != "") %>%
  select(rxcode, value)

## Makes empty array for PCH patients
## Row = 1 encounter, column = 1 medication
## Will be filled in as code goes medication by medication (see below) to see for each encounter if that pateint was administered the medication

drug_mat <- matrix(0, nrow = nenctr, ncol = length(drug_list))

## Go through all encounters one at a tinme. Within an encounter go medication by medication. 
## If the patient was administered the medication during that encounter, set value for that medication to '1', if no set value to '0'
for (i in 1:nenctr) {
  tmp_meds <-  ih_meds %>%
    filter(PATIENT_ID == ih_dat$PATIENT_ID[i] & ENCNTR_SEQ == ih_dat$ENCNTR_SEQ[i])
  
  ndrug <- nrow(tmp_meds) 
  for (j in 1:ndrug) {
    drug_id <- which(drug_tab$value == tmp_meds$DRUG_NM[j])
    nid <- length(drug_id)
    if (nid > 0) {
      for (k in 1:nid) {
        mat_id <- which(drug_list == paste0("RX", drug_tab$rxcode[drug_id[k]]))
        if (length(mat_id) > 0) {
          drug_mat[i, mat_id] <- 1
        } else {
          stop("HERE")
        }
      }
    }
    
  }
}

ih_out <- data.frame(ih_dat, drug_mat)

names(ih_out) <- c("encounter_id", "patient_id", "code_b", "age", "sex", 
                   drug_list)

save(ih_out, file = "ih_data_for_prediction.RData")
