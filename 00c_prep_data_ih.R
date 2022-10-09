set.seed(1234)
library(data.table)
library(stringi)
library(tidyverse)

load("data_for_ml.RData")

old_ih_patients <- read.csv("../../../ih_data/ID_patient_encounter_test.csv")

ih_patients <- read.csv("../../../ih_data/CRRT_iHD_identified_ICD10_ICD9_PCH_20210711.csv")
ih_patients <- ih_patients %>% 
  select(PATIENT_ID, ENCNTR_SEQ, ICD_5A1D70Z_FLG, ICD_5A1D90Z_FLG) %>%
  filter(ICD_5A1D70Z_FLG == 1 | ICD_5A1D90Z_FLG == 1) %>%
  mutate(sum = ICD_5A1D70Z_FLG + ICD_5A1D90Z_FLG) %>%
  filter(sum < 2) %>%
  mutate(Flag = ifelse(ICD_5A1D70Z_FLG == 1, "5A1D70Z", "5A1D90Z")) %>%
  select(-sum , -ICD_5A1D70Z_FLG, -ICD_5A1D90Z_FLG)


ih_demographics <- read.csv("../../../ih_data/IH_CRRT_PD_Demographics_20200204.csv")

ih_dat <- merge(ih_patients, ih_demographics, by = c("PATIENT_ID", "ENCNTR_SEQ"))

ih_dat <- ih_dat %>%
  select(PATIENT_ID, ENCNTR_SEQ, Flag, AGE_YRS, SEX_DSP)

ih_meds <- read.csv("../../../ih_data/IH_CRRT_PD_Medications_20200204.csv")

nenctr = nrow(ih_dat)
drug_list <- names(final_dat)[8:ncol(final_dat)]

drug_tab <- read.csv("../../../ih_data/selected_drugs_AMM.csv")

drug_tab <- pivot_longer(drug_tab, cols = name:X.24, values_drop_na = TRUE) %>%
  filter(value != "") %>%
  select(rxcode, value)

nenctr = nrow(ih_dat)

drug_mat <- matrix(0, nrow = nenctr, ncol = length(drug_list))

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
