suppressWarnings(suppressMessages({library(dplyr); library(lme4); library(broom.mixed)}))
options(warn=-1)
# derive repo root from this script's own location (reproduction/ -> repo root); read the full pinned dataset
.args <- commandArgs(FALSE); .f <- sub("^--file=", "", .args[grep("^--file=", .args)])
HERE <- if (length(.f)) normalizePath(dirname(.f)) else getwd()
REPO <- dirname(HERE)
df <- read.csv(file.path(REPO, "data", "multi_level_data", "level1_FINAL.csv"), stringsAsFactors=FALSE, check.names=FALSE)
cat("raw rows:", nrow(df), "\n")

nm <- c("(1) Corporations"="Corporations","(13) Social welfare or poor"="Social Welfare or Poor",
 "(14) State and local governments"="State and Local Governments","(16) Other"="Other",
 "(2) Trade and other business associations"="Trade and Business Associations",
 "(3) Occupational associations"="Occupational Associations","(4) Unions"="Unions",
 "(5) Education"="Education","(6) Health"="Health","(7) Public interest"="Public Interest",
 "(8) Identity groups"="Identity Groups")
df <- df %>%
  mutate(level1_ABBREVCAT = nm[level1_ABBREVCAT],
         level1_ABBREVCAT = if_else(level1_ABBREVCAT=="Corporations", NA_character_, level1_ABBREVCAT)) %>%
  filter(!is.na(level1_ABBREVCAT)) %>%
  mutate(level1_ABBREVCAT = case_when(
           level1_ABBREVCAT %in% c("Trade and Business Associations","Corporations") ~ "Business-Oriented Interests",
           level1_ABBREVCAT == "State and Local Governments" ~ "Government Interests",
           TRUE ~ "Non-business/nongovernment"),
         level1_ABBREVCAT = case_when(
           level1_ABBREVCAT=="Business-Oriented Interests" ~ "Business Interests",
           level1_ABBREVCAT=="Government Interests" ~ "Government Interests",
           TRUE ~ "Non-Business Interests"))
mm <- c("(1) Institution"="Institution","(2) Association of individuals"="Association of Individuals",
 "(3) Association of institutions"="Association of Institutions","(4) Government or association of governments"="Government or Association of Governments",
 "(5) Mixed"="Mixed","(6) Other"="Other","(9) Cant tell or DK"="Can't Tell")
df <- df %>% mutate(
  level1_MSHIP_STATUS11 = mm[level1_MSHIP_STATUS11],
  level1_MSHIP_STATUS11 = case_when(
    level1_MSHIP_STATUS11=="Association of Individuals" ~ "Association of Individuals",
    level1_MSHIP_STATUS11 %in% c("Institution","Association of Institutions","Government or Association of Governments") ~ "Association of Institutions",
    TRUE ~ "Other"))
df <- df %>% mutate(
  mention_year = as.integer(substr(as.character(level1_year_week),1,4)),
  year_before_termEnd = as.integer(!is.na(mention_year) & mention_year==(level1_termEndYear-1)),
  first_year_term = as.integer(!is.na(mention_year) & mention_year==level1_termBeginYear),
  term_status = case_when(
    is.na(first_year_term) | is.na(year_before_termEnd) ~ NA_character_,
    first_year_term==1 & year_before_termEnd==0 ~ "First Year",
    first_year_term==0 & year_before_termEnd==1 ~ "Year Before Term End",
    TRUE ~ "Other"))
most_common <- df %>% group_by(level1_org_id) %>%
  summarize(most_common_issue_area = { tbl<-table(level1_issue_area); if(length(tbl)==0) NA_character_ else names(tbl)[which.max(tbl)] }, .groups="drop")
unique_areas <- df %>% group_by(level1_org_id) %>% summarize(unique_issue_areas=n_distinct(level1_issue_area), .groups="drop")
df <- df %>% left_join(most_common,by="level1_org_id") %>% left_join(unique_areas,by="level1_org_id") %>%
  mutate(issue_area_overlap = as.integer(level1_issue_area==most_common_issue_area))
df <- df %>% mutate(
  saliency_measure = as.numeric(level1_issue_area_salience),
  saliency_category = case_when(is.na(saliency_measure)~NA_character_, saliency_measure<=7~"low", saliency_measure<=14~"medium", TRUE~"high"),
  saliency_category = factor(saliency_category, levels=c("low","medium","high")))
df <- df %>% mutate(
  across(c(level1_chamber_x, level1_partyHistory), ~na_if(., "")),
  level1_chamber_x=factor(level1_chamber_x), level1_partyHistory=factor(level1_partyHistory),
  level1_issue_area=factor(level1_issue_area), level1_ABBREVCAT=factor(level1_ABBREVCAT),
  level1_MSHIP_STATUS11=factor(level1_MSHIP_STATUS11), term_status=factor(term_status),
  level1_prominence=as.numeric(level1_prominence))
cat("preprocessed rows:", nrow(df), "\n")

ctl <- glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=100000))
gf  <- function(form, d) glmer(as.formula(form), data=d, family=binomial("logit"), control=ctl, nAGQ=0)
dn  <- function(d, cols) d[stats::complete.cases(d[, cols]), ]

empty <- gf("level1_prominence ~ 1 + (1|level1_org_id) + (1|level1_issue_area)", df)
vc <- as.data.frame(VarCorr(empty))
v_org <- vc$vcov[vc$grp=="level1_org_id"]; v_iss <- vc$vcov[vc$grp=="level1_issue_area"]
cat(sprintf("\nEMPTY: nobs=%d  Var(org)=%.3f  Var(issue)=%.3f  ICC_org=%.4f  ICC_issue=%.4f\n",
    nobs(empty), v_org, v_iss, v_org/(v_org+v_iss+pi^2/3), v_iss/(v_org+v_iss+pi^2/3)))

df <- df %>% mutate(saliency_category=relevel(factor(saliency_category), ref="low"),
  level1_chamber_x=relevel(factor(level1_chamber_x), ref="House of Representatives"),
  level1_partyHistory=relevel(factor(level1_partyHistory), ref="Democrat"),
  level1_ABBREVCAT=relevel(factor(level1_ABBREVCAT), ref="Business Interests"),
  level1_MSHIP_STATUS11=relevel(factor(level1_MSHIP_STATUS11), ref="Association of Institutions"),
  term_status=relevel(factor(term_status), ref="First Year"))

a1 <- gf("level1_prominence ~ saliency_category + (1|level1_org_id) + (1|level1_issue_area)", df)
a2 <- gf("level1_prominence ~ saliency_category + level1_chamber_x + level1_partyHistory + level1_MSHIP_STATUS11 + level1_ABBREVCAT + (1|level1_org_id) + (1|level1_issue_area)",
         dn(df, c("saliency_category","level1_chamber_x","level1_partyHistory","level1_MSHIP_STATUS11","level1_ABBREVCAT")))
b1 <- gf("level1_prominence ~ level1_issue_maximal_overlap + term_status + level1_bills_sponsored + level1_seniority + (1|level1_org_id) + (1|level1_issue_area)",
         dn(df, c("level1_issue_maximal_overlap","term_status","level1_bills_sponsored","level1_seniority")))
b2 <- gf("level1_prominence ~ level1_issue_maximal_overlap + term_status + level1_bills_sponsored + level1_seniority + level1_chamber_x + level1_partyHistory + level1_MSHIP_STATUS11 + level1_ABBREVCAT + (1|level1_org_id) + (1|level1_issue_area)",
         dn(df, c("level1_issue_maximal_overlap","term_status","level1_bills_sponsored","level1_seniority","level1_chamber_x","level1_partyHistory","level1_MSHIP_STATUS11","level1_ABBREVCAT")))
c1 <- gf("level1_prominence ~ level1_YEARS_EXISTED + level1_OUTSIDE11 + unique_issue_areas + (1|level1_org_id) + (1|level1_issue_area)",
         dn(df, c("level1_YEARS_EXISTED","level1_OUTSIDE11","unique_issue_areas")))
c2 <- gf("level1_prominence ~ level1_YEARS_EXISTED + level1_OUTSIDE11 + unique_issue_areas + level1_chamber_x + level1_partyHistory + level1_MSHIP_STATUS11 + level1_ABBREVCAT + (1|level1_org_id) + (1|level1_issue_area)",
         dn(df, c("level1_YEARS_EXISTED","level1_OUTSIDE11","unique_issue_areas","level1_chamber_x","level1_partyHistory","level1_MSHIP_STATUS11","level1_ABBREVCAT")))

tid <- function(m,nm){ s<-summary(m)$coefficients; data.frame(model=nm, term=rownames(s), estimate=round(s[,1],4), se=round(s[,2],4), z=round(s[,3],3), p=signif(s[,4],3), nobs=nobs(m), row.names=NULL) }
res <- do.call(rbind, list(tid(a1,"A1"),tid(a2,"A2"),tid(b1,"B1"),tid(b2,"B2"),tid(c1,"C1"),tid(c2,"C2")))
write.csv(res, file.path(HERE, "_glmer_results.csv"), row.names=FALSE)
cat("\n=== A1 (saliency) ===\n"); print(tid(a1,"A1")[,c("term","estimate","se","z")])
cat("\n=== A2 (full) ===\n"); print(tid(a2,"A2")[,c("term","estimate","se","z")])
cat("\nnobs: a1=",nobs(a1)," a2=",nobs(a2)," b1=",nobs(b1)," b2=",nobs(b2)," c1=",nobs(c1)," c2=",nobs(c2),"\n")
cat("wrote _glmer_results.csv (",nrow(res),"coefficients )\n")
