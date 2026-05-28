# ============================================================
# Machine Learning for Alzheimer's Disease Classification and Progression

# Goal:
# 1. Classify baseline diagnosis stage: CN vs MCI vs AD
# 2. Predict whether baseline MCI patients convert to AD within 3 years
# ============================================================

library(ADNIMERGE2)
library(dplyr)
library(tidyr)
library(lubridate)
library(nnet)
library(randomForest)
library(caret)
library(pROC)

# ----------------------------
# 1. Classification dataset
# ----------------------------

# ADSL is the subject-level ADNI dataset.
# Restrict to enrolled participants because screening failures/non-enrolled subjects should not be used in the modeling dataset.

classification_data <- ADSL %>%
  filter(ENRLFL == "Y") %>%
  transmute(
    # Keep subject identifiers, extract numeric RID from USUBJID so it can later be joined to DXSUM
    USUBJID,
    RID = as.numeric(sub(".*-", "", USUBJID)),
    
    # Demographic predictors
    AGE,
    SEX = factor(SEX),
    EDUC,
    # Clean diagnosis categories into 3 consistent groups:
    # CN = cognitively normal
    # MCI = mild cognitive impairment
    # AD = Alzheimer’s disease / dementia
    DX = factor(
      case_when(
        DX %in% c("CN", "Cognitively Normal") ~ "CN",
        DX %in% c("MCI", "LMCI", "EMCI") ~ "MCI",
        DX %in% c("AD", "DEM", "Dementia") ~ "AD",
        TRUE ~ NA_character_
      ),
      levels = c("CN", "MCI", "AD")
    ),
    
    # Cognitive test predictors
    MMSCORE,   # MMSE score: lower values indicate worse cognition
    ADASTT13,  # ADAS-Cog 13: higher values indicate worse cognition
    CDRSB      # Clinical Dementia Rating Sum of Boxes: higher = worse impairment
  ) %>%
  filter(!is.na(DX)) %>%
  # Ensure one row per subject for baseline classification
  distinct(USUBJID, .keep_all = TRUE)


# Select only the variables needed for the classification model
# and remove rows with missing predictor or outcome values.

classification_model_data <- classification_data %>%
  select(DX, AGE, SEX, EDUC, MMSCORE, ADASTT13, CDRSB) %>%
  drop_na()


# Set seed so the train/test split is reproducible.
set.seed(42)

# Randomly sample 70% of rows for training.
class_train_idx <- sample(
  seq_len(nrow(classification_model_data)),
  size = 0.7 * nrow(classification_model_data)
)

class_train <- classification_model_data[class_train_idx, ]
class_test  <- classification_model_data[-class_train_idx, ]


# Fit a multinomial logistic regression model.
# This is used because the outcome has 3 categories: CN, MCI, and AD.

stage_model <- multinom(
  DX ~ AGE + SEX + EDUC + MMSCORE + ADASTT13 + CDRSB,
  data = class_train
)

summary(stage_model)


# Predict diagnosis class on the held-out test set.
stage_preds <- predict(stage_model, newdata = class_test)


# Confusion matrix compares predicted diagnosis to actual diagnosis.
conf_mat <- table(
  Predicted = stage_preds,
  Actual = class_test$DX
)

conf_mat

# Overall accuracy:
# number of correct predictions divided by total predictions.
accuracy <- sum(diag(conf_mat)) / sum(conf_mat)
accuracy


# Per-class accuracy:
# shows how well the model performs separately for CN, MCI, and AD.
class_accuracy <- diag(conf_mat) / colSums(conf_mat)
class_accuracy


# caret gives a fuller evaluation:
# accuracy, sensitivity, specificity, precision, recall, and F1.
confusionMatrix(stage_preds, class_test$DX)


# Predict class probabilities instead of just class labels.
# These probabilities are needed for ROC and AUC analysis.
probs <- predict(stage_model, newdata = class_test, type = "probs")


# One-vs-rest ROC curves:
# Each diagnosis is treated as the positive class against the other two.
roc_cn  <- roc(class_test$DX == "CN",  probs[, "CN"])
roc_mci <- roc(class_test$DX == "MCI", probs[, "MCI"])
roc_ad  <- roc(class_test$DX == "AD",  probs[, "AD"])

auc(roc_cn)
auc(roc_mci)
auc(roc_ad)


# ----------------------------
# 2. Baseline MCI cohort
# ----------------------------

# For the second modeling task, I only want participants who start as MCI.
# The goal is to predict whether these MCI patients progress to AD within 3 years.

baseline_mci <- ADSL %>%
  filter(ENRLFL == "Y") %>%
  transmute(
    USUBJID,
    RID = as.numeric(sub(".*-", "", USUBJID)),
    AGE,
    SEX = factor(SEX),
    EDUC,
    DX,
    MMSCORE,
    ADASTT13,
    CDRSB
  ) %>%
  filter(DX %in% c("MCI", "LMCI", "EMCI")) %>%
  
  # Keep one baseline row per subject.
  distinct(USUBJID, .keep_all = TRUE)


# ----------------------------
# 3. Get true baseline dates from DXSUM
# ----------------------------

# DXSUM is the longitudinal diagnosis table.
# I use it to find each participant’s earliest diagnosis visit date.
# This becomes their baseline date for follow-up calculations.

baseline_dates <- DXSUM %>%
  mutate(EXAMDATE = as.Date(EXAMDATE)) %>%
  filter(!is.na(EXAMDATE)) %>%
  group_by(RID) %>%
  summarise(
    BASELINE_DATE = min(EXAMDATE),
    .groups = "drop"
  )

# Join baseline dates onto the baseline MCI cohort.
baseline_mci2 <- baseline_mci %>%
  inner_join(baseline_dates, by = "RID")


# ----------------------------
# 4. Clean longitudinal diagnosis table
# ----------------------------

# DXSUM contains diagnosis indicators across multiple visits.
# I convert those indicators into one clean diagnosis variable per visit.

dx_long <- DXSUM %>%
  mutate(
    EXAMDATE = as.Date(EXAMDATE),
    
    # Create a simplified visit-level diagnosis.
    DX_VISIT = case_when(
      DXAD   == 1 | DXAD   == "Yes" ~ "AD",
      DXMCI  == 1 | DXMCI  == "Yes" ~ "MCI",
      DXNORM == 1 | DXNORM == "Yes" ~ "CN",
      TRUE ~ NA_character_
    )
  ) %>%
  select(RID, VISCODE, EXAMDATE, DX_VISIT) %>%
  distinct()


# ----------------------------
# 5. Join baseline MCI subjects to longitudinal diagnoses
# ----------------------------

# This creates a longitudinal follow-up table for the baseline MCI cohort.
# Each subject can have multiple follow-up visits.
#Essentially combine everything so far

mci_followup <- baseline_mci2 %>%
  inner_join(dx_long, by = "RID") %>%
  mutate(
    # Calculate time from baseline visit to each follow-up visit in years.
    years_from_baseline = as.numeric(EXAMDATE - BASELINE_DATE) / 365.25
  )


# ----------------------------
# 6. Define conversion within 3 years
# ----------------------------

# For each baseline MCI subject, I check whether they had any AD diagnosis
# within 3 years after baseline.

conversion_outcome <- mci_followup %>%
  filter(
    !is.na(years_from_baseline),
    years_from_baseline > 0,
    years_from_baseline <= 3
  ) %>%
  group_by(USUBJID, RID) %>%
  summarise(
    converted_to_ad = as.integer(any(DX_VISIT == "AD")),
    .groups = "drop"
  )


# ----------------------------
# 7. Require enough follow-up for non-converters
# ----------------------------

# Important data-quality step:
# If someone never converts, I only label them as a true non-converter
# if they were observed for at least 3 years.

followup_check <- mci_followup %>%
  group_by(USUBJID, RID) %>%
  summarise(
    max_followup_years = max(years_from_baseline, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    has_3yr_followup = max_followup_years >= 3
  )


# ----------------------------
# 8. Final progression dataset
# ----------------------------

# Combine baseline predictors, conversion outcome, and follow-up eligibility.
# Keep:
# 1. People who converted to AD within 3 years
# 2. People who did not convert but had enough follow-up to confirm non-conversion

progression_data <- baseline_mci2 %>%
  left_join(conversion_outcome, by = c("USUBJID", "RID")) %>%
  left_join(followup_check, by = c("USUBJID", "RID")) %>%
  filter(converted_to_ad == 1 | has_3yr_followup) %>%
  mutate(
    # Missing conversion means no AD diagnosis within 3 years.
    converted_to_ad = ifelse(is.na(converted_to_ad), 0L, converted_to_ad),
    
    # Convert outcome to factor for classification models.
    converted_to_ad = factor(
      converted_to_ad,
      levels = c(0, 1),
      labels = c("No", "Yes")
    )
  )


# ----------------------------
# 9. Model-ready progression dataset
# ----------------------------

# Select baseline predictors and conversion outcome.
# These are the same clinically meaningful variables used earlier.

progression_model_data <- progression_data %>%
  select(
    converted_to_ad,
    AGE,
    SEX,
    EDUC,
    MMSCORE,
    ADASTT13,
    CDRSB
  ) %>%
  drop_na()


# Split into 70% training and 30% testing.
set.seed(42)

train_idx <- sample(
  seq_len(nrow(progression_model_data)),
  size = 0.7 * nrow(progression_model_data)
)

train_data <- progression_model_data[train_idx, ]
test_data  <- progression_model_data[-train_idx, ]


# ----------------------------
# 10. Logistic regression
# ----------------------------

# Logistic regression predicts probability of conversion to AD.
# This is a good baseline model because the outcome is binary: Yes vs No.

logit_fit <- glm(
  converted_to_ad ~ AGE + SEX + EDUC + MMSCORE + ADASTT13 + CDRSB,
  data = train_data,
  family = binomial()
)

summary(logit_fit)


# Predict probabilities on the test set.
pred_probs <- predict(logit_fit, newdata = test_data, type = "response")

# Evaluate using ROC and AUC.
# AUC measures how well the model separates converters from non-converters.
roc_obj <- roc(test_data$converted_to_ad, pred_probs)

auc(roc_obj)
plot(roc_obj)


# ----------------------------
# 11. Random Forest
# ----------------------------

# Random forest is a non-linear ensemble model.
# It can capture interactions and non-linear relationships between predictors.

# Check class imbalance.
prop.table(table(train_data$converted_to_ad))

# Because there are fewer converters than non-converters,
# I apply class weights so the model pays more attention to the minority class.
weights <- c(
  "No"  = 1,
  "Yes" = 4.07
)

set.seed(42)

rf_model_balanced <- randomForest(
  converted_to_ad ~ AGE + SEX + EDUC + MMSCORE + ADASTT13 + CDRSB,
  data = train_data,
  ntree = 500,
  importance = TRUE,
  classwt = weights
)

rf_model_balanced


# Predict probability of conversion.
rf_probs <- predict(rf_model_balanced, newdata = test_data, type = "prob")[,2]

roc_rf <- roc(test_data$converted_to_ad, rf_probs)

auc(roc_rf)
plot(roc_rf, col = "blue")


# ----------------------------
# 12. XGBoost
# ----------------------------

# XGBoost requires numeric matrices rather than regular data frames.
# So I convert the outcome to 0/1 and convert predictors using model.matrix.

train_label <- ifelse(train_data$converted_to_ad == "Yes", 1, 0)
test_label  <- ifelse(test_data$converted_to_ad == "Yes", 1, 0)

train_matrix <- model.matrix(
  converted_to_ad ~ . -1,
  data = train_data
)

test_matrix <- model.matrix(
  converted_to_ad ~ . -1,
  data = test_data
)


# Train XGBoost model.
# objective = binary:logistic because this is a binary classification problem.
# eval_metric = auc because I am comparing models using AUC.

set.seed(42)

xgb_model <- xgboost(
  data = train_matrix,
  label = train_label,
  nrounds = 100,
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 4,
  eta = 0.1,
  verbose = 0
)


# Predict probabilities and calculate AUC.
xgb_probs <- predict(xgb_model, newdata = test_matrix)

roc_xgb <- roc(test_data$converted_to_ad, xgb_probs)

auc(roc_xgb)
plot(roc_xgb, col = "red")


# ----------------------------
# 13. Compare all models
# ---------------------------- 

# Plot all ROC curves together.
# The model with the curve closest to the top-left corner has better discrimination.

plot(roc_obj, col = "black", main = "Model Comparison")
plot(roc_rf, col = "blue", add = TRUE)
plot(roc_xgb, col = "red", add = TRUE)


auc_logit <- auc(roc_obj)
auc_rf    <- auc(roc_rf)
auc_xgb   <- auc(roc_xgb)

legend("bottomright",
       legend = c(
         sprintf("Logistic (AUC=%.3f)", auc_logit),
         sprintf("RF (AUC=%.3f)", auc_rf),
         sprintf("XGB (AUC=%.3f)", auc_xgb)
       ),
       col = c("black", "blue", "red"),
       lwd = 2,
       cex = 0.85,     # smaller text
       bty = "n",      # remove box (cleaner)
       inset = 0.02)   # move slightly inward

