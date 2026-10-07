import streamlit as st
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from datetime import date

st.set_page_config(page_title="NobelLit 2026 Predictor v2", page_icon="🏆", layout="wide")

BASE = Path(__file__).parent
CANDIDATE_FILE = BASE / "candidates_2026.csv"
API_URL = "https://api.nobelprize.org/2.1/nobelPrizes"

@st.cache_data(ttl=86400)
def load_candidates():
    return pd.read_csv(CANDIDATE_FILE)

@st.cache_data(ttl=86400)
def load_historical_literature():
    """Fetch authoritative Nobel Literature laureate data from NobelPrize.org API."""
    params = {
        "nobelPrizeCategory": "lit",
        "limit": 200,
        "sort": "desc"
    }
    r = requests.get(API_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    rows=[]
    for prize in data.get("nobelPrizes", []):
        year = int(prize["awardYear"])
        for la in prize.get("laureates", []):
            rows.append({
                "Year": year,
                "Laureate_ID": la.get("id"),
                "Name": la.get("fullName", {}).get("en") if isinstance(la.get("fullName"),dict) else la.get("fullName"),
                "Motivation": la.get("motivation",{}).get("en") if isinstance(la.get("motivation"),dict) else "",
                "Share": la.get("portion")
            })
    return pd.DataFrame(rows)

@st.cache_data(ttl=86400)
def load_laureates_details():
    """Fetch laureate details for birth year/country where available."""
    # The API endpoint can return all laureates. We only need a manageable subset.
    r=requests.get("https://api.nobelprize.org/2.1/laureates", params={"limit":1000}, timeout=30)
    r.raise_for_status()
    data=r.json()
    rows=[]
    for la in data.get("laureates", []):
        if la.get("id") is None:
            continue
        if not la.get("nobelPrizes"):
            continue
        # Find literature prizes only
        lit=[p for p in la.get("nobelPrizes",[]) if p.get("category",{}).get("en")=="Literature"]
        if not lit:
            continue
        birth=la.get("birth",{})
        rows.append({
            "Laureate_ID":la.get("id"),
            "Name":la.get("fullName",{}).get("en",""),
            "Birth_Date":birth.get("date"),
            "Birth_City":(birth.get("place") or {}).get("city",{}).get("en") if isinstance(birth.get("place"),dict) else None,
            "Birth_Country":(birth.get("place") or {}).get("country",{}).get("en") if isinstance(birth.get("place"),dict) else None,
            "Gender":la.get("gender"),
            "Wikipedia":la.get("wikipedia"),
        })
    return pd.DataFrame(rows)

def safe_historical():
    try:
        return load_historical_literature(), None
    except Exception as e:
        return pd.DataFrame(), str(e)

def safe_details():
    try:
        return load_laureates_details(), None
    except Exception as e:
        return pd.DataFrame(), str(e)

def softmax(x):
    x=np.asarray(x,dtype=float)
    e=np.exp((x-x.max())/12)
    return e/e.sum()*100

cand=load_candidates()
hist, hist_error=safe_historical()
details, details_error=safe_details()

# Candidate scoring
profile_cols=["International_Recognition","Translation_Reach","Critical_Recognition","Career_Longevity","Major_Awards"]
cand["Literary_Profile"]=cand[profile_cols].mean(axis=1)*10
cand["Recent_Attention_Score"]=cand["Recent_Attention"]*10

st.markdown("""
# 🏆 NobelLit 2026 Predictor — Version 2
### Historical patterns + public market signals + explainable scoring
""")

st.warning(
    "PRE-ANNOUNCEMENT EDUCATIONAL MODEL: This app does not know the Swedish Academy's confidential shortlist. "
    "Market odds are public speculation. The historical analysis describes patterns among past laureates; "
    "it does not establish causal rules for who wins."
)

# Sidebar
st.sidebar.header("⚙️ Model controls")
market_w=st.sidebar.slider("Public market signal",0,100,45)
history_w=st.sidebar.slider("Historical similarity",0,100,35)
profile_w=st.sidebar.slider("Literary profile",0,100,20)
total=market_w+history_w+profile_w
if total==0:
    st.error("Set at least one weight above zero.")
    st.stop()
mw,hw,pw=[x/total for x in (market_w,history_w,profile_w)]
st.sidebar.caption(f"Normalised: Market {mw:.0%} | Historical {hw:.0%} | Profile {pw:.0%}")

tabs=st.tabs(["🏆 Prediction","📚 Historical Nobel","🔎 Author Similarity","📊 Odds Lab","🧪 Model Sandbox","ℹ️ Methodology"])

with tabs[0]:
    # Historical score: transparent heuristic based on a few dimensions.
    # No fake "trained probability".
    current_year=2026
    # Age similarity: past literature laureates' median age.
    age_hist=None
    if not hist.empty and not details.empty:
        h=hist.merge(details,on="Laureate_ID",how="left")
        h["Birth_Year_Num"]=pd.to_datetime(h["Birth_Date"],errors="coerce").dt.year
        h["Age_at_Award"]=h["Year"]-h["Birth_Year_Num"]
        age_hist=h["Age_at_Award"].dropna()
    median_age=float(age_hist.median()) if age_hist is not None and len(age_hist)>0 else 65
    # Historical score: proximity to median laureate age + small recognition adjustment.
    cand["Age_2026"]=2026-cand["Birth_Year"]
    cand["Age_Similarity"]=np.exp(-((cand["Age_2026"]-median_age)/18)**2)*100
    cand["Historical_Similarity"]=0.7*cand["Age_Similarity"]+0.3*cand["Literary_Profile"]
    cand["Combined_Score"]=mw*cand["Market_Share_%"]+hw*cand["Historical_Similarity"]+pw*cand["Literary_Profile"]
    cand["Model_Probability_%"]=softmax(cand["Combined_Score"])
    rank=cand.sort_values("Model_Probability_%",ascending=False).reset_index(drop=True)

    top=rank.iloc[0]
    a,b,c,d=st.columns(4)
    a.metric("Model favourite",top["Author"])
    b.metric("Model score",f"{top['Combined_Score']:.1f}/100")
    c.metric("Model probability-like share",f"{top['Model_Probability_%']:.1f}%")
    d.metric("Age in 2026",f"{int(top['Age_2026'])}")

    st.subheader("Prediction ranking")
    out=rank[["Author","Country","Market_Share_%","Historical_Similarity","Literary_Profile","Combined_Score","Model_Probability_%"]].copy()
    out.columns=["Author","Country","Market Share %","Historical Similarity","Literary Profile","Combined Score","Model Share %"]
    st.dataframe(out.style.format({
        "Market Share %":"{:.1f}%",
        "Historical Similarity":"{:.1f}",
        "Literary Profile":"{:.1f}",
        "Combined Score":"{:.1f}",
        "Model Share %":"{:.1f}%"
    }),use_container_width=True,hide_index=True)
    st.bar_chart(rank.set_index("Author")["Model_Probability_%"])
    st.caption("The model share is a score-derived distribution, not an official Nobel probability.")

with tabs[1]:
    st.subheader("📚 What can the official Nobel data tell us?")
    if hist_error:
        st.error(f"Could not reach NobelPrize.org right now: {hist_error}")
        st.info("Try again later. The app is designed to retrieve the official data automatically.")
    else:
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Historical prize records",len(hist))
        c2.metric("Years covered",f"{hist['Year'].min()}–{hist['Year'].max()}")
        c3.metric("Unique laureates",hist["Name"].nunique())
        if "Share" in hist:
            c4.metric("Shared-award records",(hist["Share"].astype(str)!="1").sum())

        st.subheader("Awards by decade")
        decade=(hist.assign(Decade=(hist["Year"]//10)*10)
                .groupby("Decade").size().rename("Laureate awards"))
        st.bar_chart(decade)

        st.subheader("Recent Nobel Literature laureates")
        recent=hist.sort_values("Year",ascending=False).head(20)
        st.dataframe(recent,use_container_width=True,hide_index=True)

        if not details.empty:
            h=hist.merge(details,on="Laureate_ID",how="left")
            h["Birth_Year_Num"]=pd.to_datetime(h["Birth_Date"],errors="coerce").dt.year
            h["Age_at_Award"]=h["Year"]-h["Birth_Year_Num"]
            st.subheader("Age at award")
            st.metric("Median age",f"{h['Age_at_Award'].median():.0f} years")
            st.metric("Mean age",f"{h['Age_at_Award'].mean():.1f} years")
            st.line_chart(h.groupby("Year")["Age_at_Award"].mean())
        st.markdown("Source: Nobel Prize official open data/API.")

with tabs[2]:
    st.subheader("🔎 Which historical pattern is closest to a candidate?")
    selected=st.selectbox("Choose an author",rank["Author"].tolist())
    row=rank[rank["Author"]==selected].iloc[0]

    a,b,c,d=st.columns(4)
    a.metric("Age",f"{int(row['Age_2026'])}")
    b.metric("Historical similarity",f"{row['Historical_Similarity']:.1f}/100")
    c.metric("Market share",f"{row['Market_Share_%']:.1f}%")
    d.metric("Profile",f"{row['Literary_Profile']:.1f}/100")

    st.markdown("### Candidate profile")
    features=pd.DataFrame({
        "Feature":["International recognition","Translation reach","Critical recognition","Career longevity","Major awards","Recent attention"],
        "Score":[row[x] for x in profile_cols+["Recent_Attention"]]
    }).set_index("Feature")
    st.bar_chart(features)

    if not hist.empty:
        st.markdown("### Nearest historical laureates by age")
        if not details.empty:
            h=hist.merge(details,on="Laureate_ID",how="left")
            h["Birth_Year_Num"]=pd.to_datetime(h["Birth_Date"],errors="coerce").dt.year
            h["Age_at_Award"]=h["Year"]-h["Birth_Year_Num"]
            h=h.dropna(subset=["Age_at_Award"]).copy()
            h["Age_Difference"]=abs(h["Age_at_Award"]-row["Age_2026"])
            st.dataframe(
                h.sort_values("Age_Difference")[["Year","Name","Age_at_Award","Motivation"]].head(10),
                use_container_width=True,hide_index=True
            )
        else:
            st.info("Laureate detail endpoint unavailable; age similarity is not shown.")

with tabs[3]:
    st.subheader("📊 Odds Lab")
    author=st.selectbox("Select author",cand["Author"].tolist(),key="odds_author")
    r=cand[cand["Author"]==author].iloc[0]
    fractional=float(r["Fractional_Odds_Denominator"])
    p_raw=1/(fractional+1)
    odds=p_raw/(1-p_raw)
    x,y,z=st.columns(3)
    x.metric("Bookmaker fractional odds",f"{fractional:.0f}/1")
    y.metric("Implied probability",f"{p_raw*100:.1f}%")
    z.metric("Odds",f"{odds:.2f}:1")

    st.latex(r"\text{Implied probability}=\frac{1}{\text{fractional odds}+1}")
    st.latex(r"\text{Odds}=\frac{p}{1-p}")

    st.write(
        f"For **{author}**, {fractional:.0f}/1 fractional odds correspond to about "
        f"**{p_raw*100:.1f}% implied probability** before accounting for the bookmaker's margin."
    )
    st.subheader("Public odds snapshot")
    odds_table=cand[["Author","Fractional_Odds_Denominator","Implied_Probability_Raw_%","Market_Share_%"]].sort_values("Implied_Probability_Raw_%",ascending=False)
    odds_table.columns=["Author","Fractional odds denominator","Raw implied probability %","Normalised market share %"]
    st.dataframe(odds_table.style.format({
        "Raw implied probability %":"{:.1f}%",
        "Normalised market share %":"{:.1f}%"
    }),use_container_width=True,hide_index=True)

with tabs[4]:
    st.subheader("🧪 Classroom model sandbox")
    st.write("This page lets students see how changing one feature changes the transparent score.")
    author=st.selectbox("Author",cand["Author"].tolist(),key="sandbox_author")
    r=cand[cand["Author"]==author].iloc[0]

    sliders={}
    for feature in profile_cols+["Recent_Attention"]:
        sliders[feature]=st.slider(feature.replace("_"," "),0,10,int(r[feature]),key=f"sandbox_{feature}_{author}")

    custom_profile=np.mean([sliders[x] for x in profile_cols])*10
    custom_hist=0.7*r["Age_Similarity"]+0.3*custom_profile
    custom_score=mw*r["Market_Share_%"]+hw*custom_hist+pw*custom_profile

    a,b,c=st.columns(3)
    a.metric("Custom literary profile",f"{custom_profile:.1f}")
    b.metric("Custom historical score",f"{custom_hist:.1f}")
    c.metric("Custom combined score",f"{custom_score:.1f}")

    st.markdown("""
### Teaching questions

1. What happens when the market weight increases?
2. What happens when historical similarity is given 70% weight?
3. Does a higher score always mean a higher real-world probability?
4. Why can we not say that this model has *learned* who will win?
5. What would a genuine supervised-learning dataset need?
""")

with tabs[5]:
    st.subheader("ℹ️ Methodology and limitations")
    st.markdown("""
### Data

**Historical data:** Nobel Prize official open API, covering Nobel Prize records since 1901.

**2026 candidate snapshot:** public bookmaker/prediction-market information available before the 8 October 2026 announcement.

### Why this is not ordinary logistic regression

A valid supervised classification dataset would need many candidate observations with a trustworthy label such as:

`1 = genuinely considered and won`  
`0 = genuinely considered but did not win`

The Swedish Academy's shortlist is confidential. Therefore, assigning `0` to random authors would create artificial labels.

### What Version 2 does instead

**Market signal → Historical similarity → Literary profile → weighted score → score-derived distribution**

This makes the model transparent and suitable for teaching.

### Important

Historical patterns are descriptive. They do not imply that the Nobel Academy follows a fixed formula.

Public odds are speculative and may differ substantially between markets.
""")

    st.download_button(
        "⬇️ Download candidate/model data",
        rank.to_csv(index=False).encode("utf-8"),
        "nobellit_2026_v2_results.csv",
        "text/csv"
    )

st.markdown("---")
st.caption("NobelLit 2026 Predictor v2 • Educational use • Historical source: NobelPrize.org open data/API")
