"""
PHASE 1 anchor: reproduce analysis/output/model_parameters.csv from on-disk data.
Replicates 02_Statistical_Models.ipynb (preprocessing + 9 fixed-effects logistic GLMs)
on data/multi_level_data/level1_FINAL.csv, then compares coefficient-by-coefficient to
the SAVED model_parameters.csv. READ-ONLY on all inputs.
"""
import os, sys, numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
pd.set_option('display.width', 200)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)  # reproduction/ -> repo root
LEVEL1 = os.path.join(REPO, "data", "multi_level_data", "level1_FINAL.csv")
SAVED  = os.path.join(REPO, "analysis", "output", "model_parameters.csv")

# ---------- preprocessing (verbatim from 02_Statistical_Models.ipynb) ----------
def recode_abbrevcat(df):
    df = df.copy()
    name_mapping = {"(1) Corporations":"Corporations","(13) Social welfare or poor":"Social Welfare or Poor",
        "(14) State and local governments":"State and Local Governments","(16) Other":"Other",
        "(2) Trade and other business associations":"Trade and Business Associations",
        "(3) Occupational associations":"Occupational Associations","(4) Unions":"Unions",
        "(5) Education":"Education","(6) Health":"Health","(7) Public interest":"Public Interest",
        "(8) Identity groups":"Identity Groups"}
    df['level1_ABBREVCAT'] = df['level1_ABBREVCAT'].map(name_mapping)
    df = df[df['level1_ABBREVCAT'] != 'Corporations'].copy()
    collapse = {"Trade and Business Associations":"Business-Oriented Interests","Corporations":"Business-Oriented Interests",
        "State and Local Governments":"Government Interests","Unions":"Non-business/nongovernment","Education":"Non-business/nongovernment",
        "Health":"Non-business/nongovernment","Social Welfare or Poor":"Non-business/nongovernment","Public Interest":"Non-business/nongovernment",
        "Identity Groups":"Non-business/nongovernment","Occupational Associations":"Non-business/nongovernment","Other":"Non-business/nongovernment"}
    df['level1_ABBREVCAT'] = df['level1_ABBREVCAT'].map(collapse)
    final = {"Business-Oriented Interests":"Business Interests","Government Interests":"Government Interests","Non-business/nongovernment":"Non-Business Interests"}
    df['level1_ABBREVCAT'] = df['level1_ABBREVCAT'].map(final).fillna('Non-Business Interests')
    df['business_interest'] = (df['level1_ABBREVCAT'] == 'Business Interests').astype(int)
    return df

def recode_membership_status(df):
    df = df.copy()
    name_mapping = {"(1) Institution":"Institution","(2) Association of individuals":"Association of Individuals",
        "(3) Association of institutions":"Association of Institutions","(4) Government or association of governments":"Government or Association of Governments",
        "(5) Mixed":"Mixed","(6) Other":"Other","(9) Cant tell or DK":"Can't Tell"}
    df['level1_MSHIP_STATUS11'] = df['level1_MSHIP_STATUS11'].map(name_mapping)
    collapse = {"Association of Individuals":"Association of Individuals","Institution":"Association of Institutions",
        "Association of Institutions":"Association of Institutions","Government or Association of Governments":"Association of Institutions",
        "Mixed":"Other","Other":"Other","Can't Tell":"Other"}
    df['level1_MSHIP_STATUS11'] = df['level1_MSHIP_STATUS11'].map(collapse).fillna('Other')
    return df

def create_term_status(df):
    df = df.copy()
    df['mention_year'] = pd.to_numeric(df['level1_year_week'].astype(str).str[:4].replace('nan', np.nan), errors='coerce')
    df['year_before_termEnd'] = ((df['mention_year'].notna()) & (df['mention_year'] == (df['level1_termEndYear'] - 1))).astype(int)
    df['first_year_term'] = ((df['mention_year'].notna()) & (df['mention_year'] == df['level1_termBeginYear'])).astype(int)
    def assign(row):
        if pd.isna(row.get('mention_year')): return np.nan
        if row['first_year_term'] == 1 and row['year_before_termEnd'] == 0: return "First Year"
        elif row['first_year_term'] == 0 and row['year_before_termEnd'] == 1: return "Year Before Term End"
        else: return "Other"
    df['term_status'] = df.apply(assign, axis=1)
    return df

def compute_issue_area_features(df):
    df = df.copy()
    mc = df.groupby('level1_org_id')['level1_issue_area'].agg(lambda x: x.mode().iloc[0] if len(x.mode())>0 else np.nan).reset_index()
    mc.columns = ['level1_org_id','most_common_issue_area']
    ua = df.groupby('level1_org_id')['level1_issue_area'].nunique().reset_index()
    ua.columns = ['level1_org_id','unique_issue_areas']
    df = df.merge(mc, on='level1_org_id', how='left').merge(ua, on='level1_org_id', how='left')
    df['issue_area_overlap'] = (df['level1_issue_area'] == df['most_common_issue_area']).astype(int)
    return df

def compute_saliency_measure(df):
    df = df.copy()
    df['saliency_measure'] = pd.to_numeric(df['level1_issue_area_salience'], errors='coerce')
    def cat(v):
        if pd.isna(v): return np.nan
        return 'low' if v<=7 else 'medium' if v<=14 else 'high'
    df['saliency_category'] = df['saliency_measure'].apply(cat)
    return df

def clean_categorical_variables(df):
    df = df.copy()
    for col in ['level1_chamber_x','level1_partyHistory']:
        if col in df.columns: df[col] = df[col].replace('', np.nan)
    for col in ['level1_chamber_x','level1_partyHistory','saliency_category','level1_issue_area','level1_ABBREVCAT','level1_MSHIP_STATUS11','term_status']:
        if col in df.columns: df[col] = pd.Categorical(df[col])
    return df

def preprocess_data(fp):
    df = pd.read_csv(fp)
    print(f"loaded {len(df):,} rows")
    df = recode_abbrevcat(df); df = recode_membership_status(df); df = create_term_status(df)
    df = compute_issue_area_features(df); df = compute_saliency_measure(df); df = clean_categorical_variables(df)
    print(f"preprocessed -> {len(df):,} rows")
    return df

def fit_glm_logistic(formula, data, name):
    res = smf.glm(formula, data, family=sm.families.Binomial()).fit()
    ci = res.conf_int()
    d = pd.DataFrame({'term':res.params.index,'estimate':res.params.values,'std_error':res.bse.values,
        'z_value':res.tvalues.values,'p_value':res.pvalues.values,'odds_ratio':np.exp(res.params.values),
        'ci_lower':ci.iloc[:,0].values,'ci_upper':ci.iloc[:,1].values})
    d['model'] = name
    return d

level1 = preprocess_data(LEVEL1)
print("prominence distribution:", dict(level1['level1_prominence'].value_counts(normalize=True).round(4)))

M = []
M.append(fit_glm_logistic("level1_prominence ~ 1", level1, "Empty Model A"))
M.append(fit_glm_logistic("level1_prominence ~ C(saliency_category, Treatment('low'))", level1, "Model A1 (Saliency)"))
M.append(fit_glm_logistic("""level1_prominence ~ C(saliency_category, Treatment('low')) + C(level1_chamber_x, Treatment('House of Representatives')) + C(level1_partyHistory, Treatment('Democrat')) + C(level1_MSHIP_STATUS11, Treatment('Association of Institutions')) + C(level1_ABBREVCAT, Treatment('Business Interests'))""",
    level1.dropna(subset=["level1_prominence","saliency_category","level1_chamber_x","level1_partyHistory","level1_MSHIP_STATUS11","level1_ABBREVCAT"]), "Model A2 (Full)"))
M.append(fit_glm_logistic("level1_prominence ~ 1", level1, "Empty Model B"))
M.append(fit_glm_logistic("""level1_prominence ~ level1_issue_maximal_overlap + C(term_status, Treatment('First Year')) + level1_bills_sponsored + level1_seniority""",
    level1.dropna(subset=["level1_prominence","level1_issue_maximal_overlap","term_status","level1_bills_sponsored","level1_seniority"]), "Model B1 (Politician Chars)"))
M.append(fit_glm_logistic("""level1_prominence ~ level1_issue_maximal_overlap + C(term_status, Treatment('First Year')) + level1_bills_sponsored + level1_seniority + C(level1_chamber_x, Treatment('House of Representatives')) + C(level1_partyHistory, Treatment('Democrat')) + C(level1_MSHIP_STATUS11, Treatment('Association of Institutions')) + C(level1_ABBREVCAT, Treatment('Business Interests'))""",
    level1.dropna(subset=["level1_prominence","level1_issue_maximal_overlap","term_status","level1_bills_sponsored","level1_seniority","level1_chamber_x","level1_partyHistory","level1_MSHIP_STATUS11","level1_ABBREVCAT"]), "Model B2 (Full)"))
M.append(fit_glm_logistic("level1_prominence ~ 1", level1, "Empty Model C"))
M.append(fit_glm_logistic("""level1_prominence ~ level1_YEARS_EXISTED + level1_OUTSIDE11 + unique_issue_areas""",
    level1.dropna(subset=["level1_prominence","level1_YEARS_EXISTED","level1_OUTSIDE11","unique_issue_areas"]), "Model C1 (Org Chars)"))
M.append(fit_glm_logistic("""level1_prominence ~ level1_YEARS_EXISTED + level1_OUTSIDE11 + unique_issue_areas + C(level1_chamber_x, Treatment('House of Representatives')) + C(level1_partyHistory, Treatment('Democrat')) + C(level1_MSHIP_STATUS11, Treatment('Association of Institutions')) + C(level1_ABBREVCAT, Treatment('Business Interests'))""",
    level1.dropna(subset=["level1_prominence","level1_YEARS_EXISTED","level1_OUTSIDE11","unique_issue_areas","level1_chamber_x","level1_partyHistory","level1_MSHIP_STATUS11","level1_ABBREVCAT"]), "Model C2 (Full)"))

repro = pd.concat(M, ignore_index=True)
saved = pd.read_csv(SAVED)
print(f"\nrepro rows={len(repro)}  saved rows={len(saved)}")

m = saved.merge(repro, on=['model','term'], suffixes=('_saved','_repro'), how='outer', indicator=True)
print("join coverage:", dict(m['_merge'].value_counts()))
both = m[m['_merge']=='both'].copy()
for c in ['estimate','std_error','z_value','p_value','odds_ratio']:
    both[c+'_absdiff'] = (both[c+'_saved'] - both[c+'_repro']).abs()
print("\n=== max abs diff (saved vs reproduced), across all matched terms ===")
for c in ['estimate','std_error','z_value','p_value','odds_ratio']:
    print(f"  {c:<11} max|Δ| = {both[c+'_absdiff'].max():.3e}   mean|Δ| = {both[c+'_absdiff'].mean():.3e}")
unmatched = m[m['_merge']!='both']
if len(unmatched):
    print("\nUNMATCHED terms:", unmatched[['model','term','_merge']].to_string(index=False))
worst = both.sort_values('estimate_absdiff', ascending=False).head(5)
print("\nlargest coefficient discrepancies:")
print(worst[['model','term','estimate_saved','estimate_repro','estimate_absdiff']].to_string(index=False))

both.to_csv(os.path.join(HERE, "_regression_compare.csv"), index=False)
maxd = both['estimate_absdiff'].max()
print(f"\nVERDICT: {'EXACT/BYTE-LEVEL' if maxd < 1e-6 else 'CLOSE' if maxd < 1e-3 else 'DIVERGENT'} "
      f"reproduction (max |Δestimate| = {maxd:.2e} over {len(both)} coefficients)")
