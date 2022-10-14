# ehr_pediatric_hemodialysis

Scripts to run code from: Medication based machine learning to identify subpopulations of pediatric hemodialysis patients in an electronic health record database Mcknite et al, Informatics in Medicine Unlocked.

Scripts run in order:

- 00: Preps ICD10 data for training
- 00b: Preps ICD9 data for prediction
- 00c: Preps IH data for additional vaidation
- 01b: Runs tuning for RF and XGB models
- 02: Runs cross-validation
- 03: Runs additional cross-validation for subpopulations
- 04: Runs validation on IH dataset
- 05: Makes figures from paper (PDP, VIP)
- 06: Creates and tests surrogate model
