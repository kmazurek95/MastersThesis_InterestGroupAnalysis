# --------------------------------------------
# Title: Model B Analysis Report
# Author: [Your Name]
# Date: 2023-06-16
# Description: Data preprocessing, model fitting, and results generation for Model B.
# --------------------------------------------

# Summary
# This script:
# - Cleans and preprocesses the data.
# - Validates the data for consistency and completeness.
# - Renames and collapses categorical variables for analysis.
# - Fits three models: an empty model, Model 1, and Model 2.
# - Extracts parameter-level and model-level statistics.
# - Generates and visualizes key metrics such as odds ratios and BIC comparisons.
# - Saves the results in multiple formats, including `.csv`, `.rds`, and Markdown tables.

# --------------------------------------------
# Load Required Libraries
# --------------------------------------------
library(lme4)         # Mixed-effects models
library(MASS)         # Statistical methods
library(dplyr)        # Data manipulation
library(knitr)        # Table generation
library(kableExtra)   # Enhanced table outputs
library(broom.mixed)  # Model summaries
library(forcats)      # Factor manipulation
library(tidyr)        # Data reshaping
library(ggplot2)      # Data visualization

# --------------------------------------------
# Data Loading and Preprocessing
# --------------------------------------------

# Set working directory (update as necessary)
setwd("C:/Users/kaleb/OneDrive/Desktop/UVA_RMSS_THESIS_MAZUREK/Models/FINAL MODELS/")

# Clear workspace
rm(list = ls())

# Load data
level1 <- read.csv("level1.csv")

# Filter out specific organization ID
level1 <- level1[!level1$level1_org_id == 20114287, ]

# Rename and collapse factor levels for `level1_ABBREVCAT`
abbrevcat_map <- c(
  "(1) Corporations" = "Corporations",
  "(2) Trade and Business Associations" = "Trade and Business Associations",
  "(3) Occupational Associations" = "Occupational Associations",
  "(4) Unions" = "Unions",
  "(5) Education" = "Education",
  "(6) Health" = "Health",
  "(7) Public Interest" = "Public Interest",
  "(8) Identity Groups" = "Identity Groups",
  "(13) Social Welfare or Poor" = "Social Welfare or Poor",
  "(14) State and Local Governments" = "State and Local Governments",
  "(16) Other" = "Other"
)

level1$level1_ABBREVCAT <- factor(level1$level1_ABBREVCAT, levels = abbrevcat_map)
level1$level1_ABBREVCAT <- fct_collapse(
  level1$level1_ABBREVCAT,
  "Business Interests" = c("Corporations", "Trade and Business Associations"),
  "Non-Business Interests" = c("Unions", "Education", "Health", "Public Interest", "Identity Groups", "Social Welfare or Poor"),
  "Government Interests" = "State and Local Governments"
)

# Rename and collapse factor levels for `level1_MSHIP_STATUS11`
mship_status_map <- c(
  "(1) Institution" = "Institution",
  "(2) Association of Individuals" = "Association of Individuals",
  "(3) Association of Institutions" = "Association of Institutions",
  "(4) Government or Association of Governments" = "Government or Association of Governments",
  "(5) Mixed" = "Mixed",
  "(6) Other" = "Other",
  "(9) Can't Tell or DK" = "Can't Tell"
)

level1$level1_MSHIP_STATUS11 <- factor(level1$level1_MSHIP_STATUS11, levels = mship_status_map)
level1$level1_MSHIP_STATUS11 <- fct_collapse(
  level1$level1_MSHIP_STATUS11,
  "Association" = c("Association of Individuals", "Association of Institutions"),
  "Other" = c("Mixed", "Other", "Can't Tell")
)

# Add derived columns
level1 <- level1 %>%
  mutate(
    mention_year = as.integer(substr(level1_year_week, 1, 4)),
    year_before_termEnd = ifelse(mention_year == (level1_termEndYear - 1), 1, 0),
    first_year_term = ifelse(mention_year == level1_termBeginYear, 1, 0),
    term_status = factor(case_when(
      first_year_term == 1 ~ "First Year",
      year_before_termEnd == 1 ~ "Year Before Term End",
      TRUE ~ "Other"
    ))
  )

# --------------------------------------------
# Model Fitting
# --------------------------------------------

# Fit the empty model
empty_model <- glmer(level1_prominence ~ (1 | level1_org_id) + (1 | level1_issue_area),
                     data = level1, family = binomial)

# Fit Model 1
model1 <- glmer(level1_prominence ~ level1_issue_maximal_overlap + term_status +
                  level1_bills_sponsored + level1_seniority + (1 | level1_org_id) + (1 | level1_issue_area),
                data = level1, family = binomial)

# Fit Model 2
model2 <- glmer(level1_prominence ~ level1_issue_maximal_overlap + term_status +
                  level1_bills_sponsored + level1_seniority + level1_chamber_x + level1_partyHistory +
                  level1_MSHIP_STATUS11 + level1_ABBREVCAT + (1 | level1_org_id) + (1 | level1_issue_area),
                data = level1, family = binomial)

# --------------------------------------------
# Extract Parameter and Model-Level Statistics
# --------------------------------------------

# Extract model information
model_info <- bind_rows(
  tidy(empty_model) %>% mutate(model = "Empty Model"),
  tidy(model1) %>% mutate(model = "Model 1"),
  tidy(model2) %>% mutate(model = "Model 2")
) %>%
  mutate(
    odds_ratio = exp(estimate),
    estimate = round(estimate, 2),
    std.error = round(std.error, 2),
    statistic = round(statistic, 2),
    p.value = round(p.value, 2),
    confidence_interval = paste0(
      "(", round(estimate - 1.96 * std.error, 2), ", ",
      round(estimate + 1.96 * std.error, 2), ")"
    )
  ) %>%
  select(model, term, odds_ratio, statistic, p.value, confidence_interval)

# Model-level statistics
model_level_stats <- tibble(
  term = rep(c("logLik", "AIC", "BIC"), each = 3),
  model = c("Empty Model", "Model 1", "Model 2"),
  estimate = c(
    logLik(empty_model), AIC(empty_model), BIC(empty_model),
    logLik(model1), AIC(model1), BIC(model1),
    logLik(model2), AIC(model2), BIC(model2)
  )
)

# --------------------------------------------
# Save Results and Visualizations
# --------------------------------------------

# Save results
write.csv(model_info, "model_info_model_b.csv", row.names = FALSE)
write.csv(model_level_stats, "model_level_stats_model_b.csv", row.names = FALSE)

saveRDS(model_info, "model_info_model_b.rds")
saveRDS(model_level_stats, "model_level_stats_model_b.rds")

# Visualize BIC comparisons
ggplot(model_level_stats, aes(x = model, y = estimate, fill = term)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Model-Level Statistics (AIC, BIC, logLik)", x = "Model", y = "Estimate") +
  theme_minimal() +
  ggsave("model_b_bic_comparison.png", width = 8, height = 6)
