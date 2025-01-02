# --------------------------------------------
# Title: Model A Analysis Report
# Author: [Your Name]
# Date: 2023-06-16
# Description: Data preprocessing, model fitting, and results generation for Model A.
# --------------------------------------------

# Summary
# This script:
# - Cleans and preprocesses the data.
# - Validates the data for integrity.
# - Fits an empty model, Model 1, and Model 2.
# - Extracts parameter-level and model-level statistics.
# - Visualizes odds ratios and BIC comparisons.
# - Saves the results to .csv, .rds, and .md files.
# - Displays the results in R Markdown.

# --------------------------------------------
# Load Required Libraries
# --------------------------------------------
library(lme4)         # For mixed-effects models
library(dplyr)        # For data manipulation
library(forcats)      # For factor manipulation
library(assertthat)   # For data validation
library(ggplot2)      # For plotting
library(knitr)        # For generating tables
library(kableExtra)   # For enhanced table outputs
library(broom.mixed)  # For extracting model summaries
library(tidyr)        # For reshaping data

# --------------------------------------------
# Data Loading and Preprocessing
# --------------------------------------------

# Load Data
level1 <- read.csv("level1.csv")

# Clean and preprocess data
level1 <- level1 %>%
  filter(level1_org_id != 20114287) %>%  # Remove rows with a specific org ID
  mutate(
    level1_ABBREVCAT = as.factor(level1_ABBREVCAT),
    level1_MSHIP_STATUS11 = as.factor(level1_MSHIP_STATUS11),
    level1_chamber_x = replace(level1_chamber_x, level1_chamber_x == "", NA),
    level1_partyHistory = replace(level1_partyHistory, level1_partyHistory == "", NA)
  )

# Validate Dataset
assert_that(
  all(c("level1_ABBREVCAT", "level1_MSHIP_STATUS11", "saliency_measure") %in% names(level1)),
  noNA(level1$level1_ABBREVCAT),
  noNA(level1$level1_MSHIP_STATUS11),
  noNA(level1$saliency_measure)
)

# Confirm expected factor levels
valid_abbrev_levels <- c("Business Interests", "Non-Business Interests", "Government Interests")
valid_status_levels <- c("Association", "Other")
assert_that(all(levels(level1$level1_ABBREVCAT) %in% valid_abbrev_levels))
assert_that(all(levels(level1$level1_MSHIP_STATUS11) %in% valid_status_levels))

# Helper function for renaming factor levels
rename_levels <- function(df, column, rename_map) {
  levels(df[[column]]) <- rename_map[levels(df[[column]])]
  df
}

# Define mappings
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

mship_status_map <- c(
  "(1) Institution" = "Institution",
  "(2) Association of Individuals" = "Association of Individuals",
  "(3) Association of Institutions" = "Association of Institutions",
  "(4) Government or Association of Governments" = "Government or Association of Governments",
  "(5) Mixed" = "Mixed",
  "(6) Other" = "Other",
  "(9) Can't Tell or DK" = "Can't Tell"
)

# Apply renaming
level1 <- rename_levels(level1, "level1_ABBREVCAT", abbrevcat_map)
level1 <- rename_levels(level1, "level1_MSHIP_STATUS11", mship_status_map)

# Collapse broader categories
level1 <- level1 %>%
  mutate(
    level1_ABBREVCAT = fct_collapse(
      level1_ABBREVCAT,
      "Business Interests" = c("Corporations", "Trade and Business Associations"),
      "Non-Business Interests" = c("Unions", "Education", "Health", "Public Interest", "Identity Groups", "Social Welfare or Poor"),
      "Government Interests" = "State and Local Governments"
    ),
    level1_MSHIP_STATUS11 = fct_collapse(
      level1_MSHIP_STATUS11,
      "Association" = c("Association of Individuals", "Association of Institutions"),
      "Other" = c("Mixed", "Other", "Can't Tell")
    )
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

# Function to fit a model and display results
fit_and_report <- function(formula, data, model_name) {
  cat("\n--- Fitting", model_name, "---\n")
  
  model <- glmer(formula, data = data, family = binomial)
  bic <- BIC(model)
  odds_ratios <- exp(fixef(model))
  
  # Display model results
  cat("BIC:", bic, "\n")
  cat("Odds Ratios:\n")
  print(odds_ratios)
  print(summary(model))
  
  list(model = model, BIC = bic, odds_ratios = odds_ratios)
}

# Fit models
model1 <- fit_and_report(level1_prominence ~ saliency_category + (1 | level1_org_id) + (1 | level1_issue_area), level1, "Model 1")
model2 <- fit_and_report(level1_prominence ~ saliency_category + level1_chamber_x + level1_partyHistory + level1_MSHIP_STATUS11 + level1_ABBREVCAT + 
                           (1 | level1_org_id) + (1 | level1_issue_area), level1, "Model 2")

# --------------------------------------------
# Extract Parameter and Model-Level Statistics
# --------------------------------------------

# Extract model information
empty_model_info <- tidy(empty_model) %>% mutate(model = "empty_model")
model1_info <- tidy(model1$model) %>% mutate(model = "model1")
model2_info <- tidy(model2$model) %>% mutate(model = "model2")

# Calculate odds ratios
empty_model_info <- empty_model_info %>% mutate(odds_ratio = exp(estimate))
model1_info <- model1_info %>% mutate(odds_ratio = exp(estimate))
model2_info <- model2_info %>% mutate(odds_ratio = exp(estimate))

# Combine parameter-level statistics
model_info <- bind_rows(empty_model_info, model1_info, model2_info) %>%
  mutate(
    estimate = round(estimate, 2),
    std.error = round(std.error, 2),
    statistic = round(statistic, 2),
    p.value = round(p.value, 2),
    odds_ratio = round(odds_ratio, 2),
    lower_ci = round(estimate - 1.96 * std.error, 2),
    upper_ci = round(estimate + 1.96 * std.error, 2),
    confidence_interval = paste0("(", lower_ci, ", ", upper_ci, ")")
  ) %>%
  select(-estimate, -std.error, -lower_ci, -upper_ci) %>%
  arrange(model, term)

# Create model-level statistics
model_level_stats <- tibble(
  term = rep(c("logLik", "AIC", "BIC"), times = 3),
  model = rep(c("empty_model", "model1", "model2"), each = 3),
  estimate = c(
    as.numeric(logLik(empty_model)), AIC(empty_model), BIC(empty_model),
    as.numeric(logLik(model1$model)), AIC(model1$model), BIC(model1$model),
    as.numeric(logLik(model2$model)), AIC(model2$model), BIC(model2$model)
  )
) %>%
  mutate(estimate = round(estimate, 2)) %>%
  arrange(model, term)

# --------------------------------------------
# Visualizations
# --------------------------------------------

# Function to plot odds ratios
plot_odds_ratios <- function(model, title, file_name) {
  odds_df <- data.frame(
    Term = names(fixef(model$model)),
    Odds_Ratio = exp(fixef(model$model))
  )
  
  ggplot(odds_df, aes(x = reorder(Term, Odds_Ratio), y = Odds_Ratio)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +
    labs(title = title, x = "Terms", y = "Odds Ratio") +
    theme_minimal() +
    ggsave(file_name, width = 8, height = 6)
}

# Create plots
plot_odds_ratios(model1, "Odds Ratios from Model 1", "model1_odds_ratios.png")
plot_odds_ratios(model2, "Odds Ratios from Model 2", "model2_odds_ratios.png")

# Compare BIC values
bic_values <- data.frame(
  Model = c("Empty Model", "Model 1", "Model 2"),
  BIC = c(BIC(empty_model), model1$BIC, model2$BIC)
)

ggplot(bic_values, aes(x = Model, y = BIC, fill = Model)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  labs(title = "BIC Comparison Across Models", x = "Model", y = "BIC") +
  theme_minimal() +
  ggsave("bic_comparison.png", width = 8, height = 6)

# --------------------------------------------
# Save Results
# --------------------------------------------

# Save parameter-level and model-level statistics
write.csv(model_info, "model_info.csv", row.names = FALSE)
write.csv(model_level_stats, "model_level_stats.csv", row.names = FALSE)

saveRDS(model_info, "model_info.rds")
saveRDS(model_level_stats, "model_level_stats.rds")
