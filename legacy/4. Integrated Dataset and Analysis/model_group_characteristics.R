# --------------------------------------------
# Title: Model C Analysis Report
# Author: [Your Name]
# Date: [Current Date]
# Description: Data preprocessing, model fitting, and results generation for Model C.
# --------------------------------------------

# Summary
# This script:
# - Cleans and preprocesses the data.
# - Validates the data for integrity.
# - Renames and collapses factor levels for categorical variables.
# - Fits three models: an empty model, Model 1, and Model 2.
# - Extracts parameter-level and model-level statistics.
# - Visualizes odds ratios and BIC comparisons.
# - Saves the results in multiple formats for detailed analysis.

# --------------------------------------------
# Load Required Libraries
# --------------------------------------------
library(lme4)         # Mixed-effects models
library(MASS)         # Statistical functions
library(dplyr)        # Data manipulation
library(knitr)        # Table generation
library(kableExtra)   # Enhanced table outputs
library(broom.mixed)  # Model summaries
library(forcats)      # Factor manipulation
library(tidyr)        # Data reshaping
library(ggplot2)      # Data visualization
library(assertthat)   # Data validation

# Set random seed for reproducibility
set.seed(123)

# --------------------------------------------
# Data Loading and Preprocessing
# --------------------------------------------

# Set working directory (update as necessary)
setwd("C:/Users/kaleb/OneDrive/Desktop/UVA_RMSS_THESIS_MAZUREK/Models/FINAL MODELS/")

# Remove existing objects to start fresh
rm(list = ls())

# Load data
level1 <- read.csv("level1.csv")

# Validate the dataset structure
assert_that(all(c("level1_ABBREVCAT", "level1_MSHIP_STATUS11", "level1_year_week") %in% names(level1)))

# Filter out rows with specific organization IDs and handle missing data
level1 <- level1 %>%
  filter(level1_org_id != 20114287) %>%
  drop_na(level1_ABBREVCAT, level1_MSHIP_STATUS11)

# Rename and collapse levels for `level1_ABBREVCAT`
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

# Rename and collapse levels for `level1_MSHIP_STATUS11`
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

# Add derived columns for analysis
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
model1 <- glmer(level1_prominence ~ level1_YEARS_EXISTED + level1_OUTSIDE11 + unique_issue_areas + 
                  (1 | level1_org_id) + (1 | level1_issue_area),
                data = level1, family = binomial)

# Fit Model 2
model2 <- glmer(level1_prominence ~ level1_YEARS_EXISTED + level1_OUTSIDE11 + unique_issue_areas +
                  level1_chamber_x + level1_partyHistory + level1_MSHIP_STATUS11 + level1_ABBREVCAT +
                  (1 | level1_org_id) + (1 | level1_issue_area),
                data = level1, family = binomial)

# --------------------------------------------
# Extract Parameter and Model-Level Statistics
# --------------------------------------------

# Combine parameter-level statistics
model_info <- bind_rows(
  tidy(empty_model) %>% mutate(model = "Empty Model"),
  tidy(model1) %>% mutate(model = "Model 1"),
  tidy(model2) %>% mutate(model = "Model 2")
) %>%
  mutate(
    odds_ratio = exp(estimate),
    confidence_interval = paste0(
      "(", round(estimate - 1.96 * std.error, 2), ", ",
      round(estimate + 1.96 * std.error, 2), ")"
    ),
    estimate = round(estimate, 2),
    std.error = round(std.error, 2),
    p.value = round(p.value, 2)
  ) %>%
  select(model, term, odds_ratio, confidence_interval, p.value)

# Create model-level statistics
model_level_stats <- tibble(
  Model = c("Empty Model", "Model 1", "Model 2"),
  BIC = c(BIC(empty_model), BIC(model1), BIC(model2)),
  AIC = c(AIC(empty_model), AIC(model1), AIC(model2)),
  logLik = c(logLik(empty_model), logLik(model1), logLik(model2))
)

# --------------------------------------------
# Visualization
# --------------------------------------------

# Odds Ratio Plot
plot_odds_ratios <- function(model_info, model_name, file_name) {
  ggplot(model_info %>% filter(model == model_name), aes(x = term, y = odds_ratio)) +
    geom_point(color = "steelblue", size = 3) +
    geom_errorbar(aes(ymin = exp(estimate - 1.96 * std.error), ymax = exp(estimate + 1.96 * std.error)), width = 0.2) +
    coord_flip() +
    labs(title = paste("Odds Ratios for", model_name), x = "Terms", y = "Odds Ratio") +
    theme_minimal() +
    ggsave(file_name, width = 8, height = 6)
}

# Generate odds ratio plots
plot_odds_ratios(model_info, "Model 1", "model1_odds_ratios.png")
plot_odds_ratios(model_info, "Model 2", "model2_odds_ratios.png")

# BIC Comparison Plot
ggplot(model_level_stats, aes(x = Model, y = BIC, fill = Model)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  labs(title = "BIC Comparison Across Models", x = "Model", y = "BIC") +
  theme_minimal() +
  ggsave("bic_comparison.png", width = 8, height = 6)

# --------------------------------------------
# Save Results
# --------------------------------------------

# Output directory
output_dir <- "results"
dir.create(output_dir, showWarnings = FALSE)

# Save results
write.csv(model_info, file.path(output_dir, "model_info_model_c.csv"), row.names = FALSE)
write.csv(model_level_stats, file.path(output_dir, "model_level_stats_model_c.csv"), row.names = FALSE)
saveRDS(model_info, file.path(output_dir, "model_info_model_c.rds"))
saveRDS(model_level_stats, file.path(output_dir, "model_level_stats_model_c.rds"))
writeLines(knitr::kable(model_info, caption = "Parameter-Level Statistics"), file.path(output_dir, "model_info.md"))
writeLines(knitr::kable(model_level_stats, caption = "Model-Level Statistics"), file.path(output_dir, "model_level_stats.md"))
