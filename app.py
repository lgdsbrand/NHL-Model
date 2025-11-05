import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NHL Score Model", page_icon="🏒", layout="centered")

# =========================
# CONFIG: where to load your Excel
# =========================
# Option A: keep the Excel in your repo at this path:
EXCEL_PATH = "data/NHL_stats.xlsx"       # GF/GA/Goalie sheets (see layout below)

# Option B: OR host the Excel and paste a raw URL here (leave None to skip)
EXCEL_URL = None  # e.g. "https://raw.githubusercontent.com/lgdsbrand/nhl-model/main/data/NHL_stats.xlsx"

# Optional: a batch games file in repo or URL (Away Team, Home Team, [Book Total])
GAMES_PATH = None
GAMES_URL  = None

# =========================
# LOADERS for your exact Excel layout (from your photo)
# =========================
def _read_excel_bytes(path_or_url):
    if path_or_url is None:
        raise ValueError("Provide EXCEL_PATH or EXCEL_URL or upload in the sidebar.")
    if isinstance(path_or_url, str) and path_or_url.startswith("http"):
        import requests
        r = requests.get(path_or_url)
        r.raise_for_status()
        return io.BytesIO(r.content)
    # local file path in repo
    return path_or_url

@st.cache_data
def load_from_layout(path_or_url):
    """
    Reads Excel with 3 sheets and returns TEAMS, GOALIES.
    Layout (by column letters):
      - GF sheet:     A=Team, C=GF/G, G=PP%, L=PK%
      - GA sheet:     B=Team, C=GP,  D=GA/G
      - Goalie sheet: A=Name, B=Team, H=GAA, L=SV%
    """
    src = _read_excel_bytes(path_or_url)

    gf = pd.read_excel(src, sheet_name="GF",     usecols="A,C,G,L", header=0).copy()
    ga = pd.read_excel(src, sheet_name="GA",     usecols="B,C,D",   header=0).copy()
    gl = pd.read_excel(src, sheet_name="Goalie", usecols="A,B,H,L", header=0).copy()

    gf.columns = ["Team", "GF/G", "PP%", "PK%"]
    ga.columns = ["Team", "GP", "GA/G"]
    gl.columns = ["Goalie", "Team", "GAA", "SV%"]

    # clean numbers
    for c in ["GF/G", "PP%", "PK%"]:
        gf[c] = pd.to_numeric(gf[c], errors="coerce")
    for c in ["GP", "GA/G"]:
        ga[c] = pd.to_numeric(ga[c], errors="coerce")
    gl["GAA"] = pd.to_numeric(gl["GAA"], errors="coerce")
    gl["SV%"] = pd.to_numeric(gl["SV%"], errors="coerce")
    gl.loc[gl["SV%"] > 1.5, "SV%"] /= 100.0  # accept 92.1 as 0.921

    teams = gf.merge(ga[["Team", "GA/G"]], on="Team", how="inner")

    TEAMS = pd.DataFrame({
        "Team":   teams["Team"].astype(str),
        "xGF_pg": teams["GF/G"],       # using GF/G as xGF proxy
        "xGA_pg": teams["GA/G"],       # using GA/G as xGA proxy
        "PP%":    teams["PP%"],
        "PK%":    teams["PK%"],
    }).dropna(subset=["Team"]).reset_index(drop=True)

    GOALIES = gl[["Team", "Goalie", "SV%", "GAA"]].copy()
    GOALIES["Team"] = GOALIES["Team"].astype(str)
    GOALIES["Goalie"] = GOALIES["Goalie"].astype(str)
    GOALIES = GOALIES.dropna(subset=["Team","Goalie"]).reset_index(drop=True)

    return TEAMS, GOALIES

def read_games_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    lower = {str(c).lower(): c for c in df_raw.columns}
    need = {"away team","home team"}
    if not need.issubset(set(lower.keys())):
        raise ValueError("Games file must include 'Away Team' and 'Home Team' columns.")
    df = df_raw.rename(columns={
        lower["away team"]:"Away Team",
        lower["home team"]:"Home Team"
    })
    if "book total" in lower:
        df = df.rename(columns={lower["book total"]:"Book Total"})
    return df

# =========================
# MODEL CORE (same as prior approved version)
# =========================
def coerce_pct(v):
    if v is None or pd.isna(v): return None
    v = float(v)
    return v/100.0 if v > 1.5 else v

def goalie_strength(sv_pct=None, gaa=None,
                    league_sv=0.910, league_gaa=3.00,
                    scale=0.15, cap=0.40):
    if scale == 0 or (sv_pct is None and gaa is None): return 0.0
    sv = None if sv_pct is None else coerce_pct(sv_pct)
    sv_delta  = 0.0 if sv  is None else (sv - league_sv) / 0.010
    gaa_delta = 0.0 if gaa is None else (league_gaa - gaa) / 1.00
    raw = 0.6*sv_delta + 0.4*gaa_delta
    return float(np.sign(raw) * min(abs(raw) * scale, cap))

def apply_special_teams(xgf_pg, opp_xga_pg, own_pp, opp_pk,
                        pp_base=20.8, pk_base=79.1, pp_w=0.06, pk_w=0.06):
    own_pp = own_pp if pd.notna(own_pp) else pp_base
    opp_pk = opp_pk if pd.notna(opp_pk) else pk_base
    off  = xgf_pg * (1 + ((own_pp - pp_base)/100.0) * pp_w)
    deff = opp_xga_pg * (1 - ((opp_pk - pk_base)/100.0) * pk_w)
    return off, deff

def expected_goals(own_xgf_pg, opp_xga_pg, opp_goalie_adj, own_pp, opp_pk,
                   team_off_w=0.50, opp_def_w=0.35, clip_low=0.3, clip_high=8.0,
                   pp_base=20.8, pk_base=79.1, pp_w=0.06, pk_w=0.06):
    off_adj, def_adj = apply_special_teams(own_xgf_pg, opp_xga_pg, own_pp, opp_pk, pp_base, pk_base, pp_w, pk_w)
    mu = team_off_w*off_adj + opp_def_w*def_adj - opp_goalie_adj
    return float(np.clip(mu, clip_low, clip_high))

def apply_inflators(away_mu, home_mu, pace_bonus=0.05, hfa_goals=0.12, clip_low=0.3, clip_high=8.0):
    away_mu *= (1 + pace_bonus)
    home_mu *= (1 + pace_bonus)
    away_mu -= hfa_goals/2
    home_mu += hfa_goals/2
    return float(np.clip(away_mu, clip_low, clip_high)), float(np.clip(home_mu, clip_low, clip_high))

def simulate_batch(games_df, params):
    rows = []
    for _, g in games_df.iterrows():
        away_gs = goalie_strength(g.get("Away SV%"), g.get("Away GAA"),
                                  params["LEAGUE_SV"], params["LEAGUE_GAA"],
                                  params["GOALIE_SCALE"], params["GOALIE_CAP"])
        home_gs = goalie_strength(g.get("Home SV%"), g.get("Home GAA"),
                                  params["LEAGUE_SV"], params["LEAGUE_GAA"],
                                  params["GOALIE_SCALE"], params["GOALIE_CAP"])

        away_mu = expected_goals(g["Away xGF_pg"], g["Home xGA_pg"], home_gs,
                                 g.get("Away PP%"), g.get("Home PK%"),
                                 params["TEAM_OFF_W"], params["OPP_DEF_W"],
                                 params["CLIP_LOW"], params["CLIP_HIGH"],
                                 params["LEAGUE_PP"], params["LEAGUE_PK"],
                                 params["PP_W"], params["PK_W"])

        home_mu = expected_goals(g["Home xGF_pg"], g["Away xGA_pg"], away_gs,
                                 g.get("Home PP%"), g.get("Away PK%"),
                                 params["TEAM_OFF_W"], params["OPP_DEF_W"],
                                 params["CLIP_LOW"], params["CLIP_HIGH"],
                                 params["LEAGUE_PP"], params["LEAGUE_PK"],
                                 params["PP_W"], params["PK_W"])

        away_mu, home_mu = apply_inflators(away_mu, home_mu,
                                           params["PACE_BONUS"], params["HFA_GOALS"],
                                           params["CLIP_LOW"], params["CLIP_HIGH"])

        sims = params["SIMS"]
        away_g = np.random.poisson(max(away_mu, 0.01), sims)
        home_g = np.random.poisson(max(home_mu, 0.01), sims)
        ties   = sims - (away_g > home_g).sum() - (home_g > away_g).sum()
        away_wp = ((away_g > home_g).sum() + 0.5*ties) / sims * 100
        home_wp = 100 - away_wp

        rows.append({
            "Away Team": g["Away Team"], "Home Team": g["Home Team"],
            "Away μ": round(float(away_mu),2), "Home μ": round(float(home_mu),2),
            "Total (Model)": round(float(away_mu + home_mu),2),
            "Away Win %": round(float(away_wp),1), "Home Win %": round(float(home_wp),1),
            "Predicted Winner": g["Away Team"] if away_wp > home_wp else g["Home Team"],
            "Book Total": g.get("Book Total", "—")
        })

    # Deterministic total calibration 4.5–6.5 based on slate average
    if params["CALIBRATE"] and rows:
        avg_total = np.mean([r["Total (Model)"] for r in rows])
        rmin, rmax = params["RANGE_MIN"], params["RANGE_MAX"]
        if avg_total < 4.8:
            target_total = rmax
        elif avg_total < 5.3:
            target_total = 6.3
        elif avg_total < 5.8:
            target_total = 6.0
        elif avg_total < 6.2:
            target_total = 5.8
        elif avg_total < 6.8:
            target_total = 5.5
        else:
            target_total = rmin
        target_total = float(np.clip(target_total, rmin, rmax))
        k = target_total / avg_total
        for r in rows:
            r["Away μ"] = round(r["Away μ"] * k, 2)
            r["Home μ"] = round(r["Home μ"] * k, 2)
            r["Total (Model)"] = round(r["Away μ"] + r["Home μ"], 2)
        st.info(f"Calibrated league total to {target_total:.2f} (range {rmin}-{rmax})")

    return pd.DataFrame(rows)

# =========================
# UI SIDEBAR (sliders)
# =========================
with st.sidebar:
    st.header("Model Settings")
    SIMS = st.slider("Simulations", 2000, 50000, 10000, 1000)
    TEAM_OFF_W = st.slider("Team Offense Weight", 0.0, 1.0, 0.50, 0.01)
    OPP_DEF_W  = st.slider("Opponent Defense Weight", 0.0, 1.0, 0.35, 0.01)
    GOALIE_SCALE = st.slider("Goalie Scale", 0.00, 0.30, 0.15, 0.01)
    GOALIE_CAP   = st.slider("Goalie Cap (goals)", 0.0, 1.0, 0.40, 0.01)
    PACE_BONUS   = st.slider("Pace Bonus", 0.00, 0.15, 0.05, 0.01)
    HFA_GOALS    = st.slider("Home-Ice Advantage (goals)", 0.00, 0.40, 0.12, 0.01)
    LEAGUE_PP    = st.number_input("League PP% baseline", value=20.8, step=0.1)
    LEAGUE_PK    = st.number_input("League PK% baseline", value=79.1, step=0.1)
    PP_W         = st.slider("PP% Weight", 0.00, 0.15, 0.06, 0.01)
    PK_W         = st.slider("PK% Weight", 0.00, 0.15, 0.06, 0.01)
    CALIBRATE    = st.checkbox("Calibrate totals to range (4.5–6.5)", value=True)
    RANGE_MIN    = 4.5
    RANGE_MAX    = 6.5
    CLIP_LOW, CLIP_HIGH = 0.3, 8.0
    LEAGUE_SV, LEAGUE_GAA = 0.910, 3.00

# =========================
# LOAD DATA: try path/URL first, else file uploader
# =========================
st.title("🏒 NHL Score Prediction Model")

TEAMS = GOALIES = None
load_err = None

try:
    src = EXCEL_URL if EXCEL_URL else EXCEL_PATH
    TEAMS, GOALIES = load_from_layout(src)
    st.success(f"✅ Loaded from {'URL' if EXCEL_URL else 'repo file'}: Teams={len(TEAMS)}, Goalies={len(GOALIES)}")
except Exception as e:
    load_err = str(e)

if TEAMS is None or GOALIES is None:
    st.warning("Could not auto-load Excel. Upload your 3-sheet Excel (GF / GA / Goalie).")
    up = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx","xls"])
    if up:
        try:
            TEAMS, GOALIES = load_from_layout(io.BytesIO(up.read()))
            st.success(f"✅ Loaded from uploaded Excel: Teams={len(TEAMS)}, Goalies={len(GOALIES)}")
        except Exception as e:
            st.error(f"❌ Upload parse error: {e}")
            st.stop()
    else:
        if load_err:
            st.caption(f"Loader hint: {load_err}")
        st.stop()

# Optional: load a batch Games file
GAMES = None
try:
    if GAMES_URL:
        GAMES = read_games_table(pd.read_csv(GAMES_URL))
    elif GAMES_PATH:
        GAMES = read_games_table(pd.read_csv(GAMES_PATH))
except Exception as e:
    st.caption(f"Games load note: {e}")

# =========================
# SINGLE GAME UI
# =========================
st.header("Single Game Prediction")

teams_list = sorted(TEAMS["Team"].unique().tolist())

c1, c2 = st.columns(2)
away_team = c1.selectbox("Away Team", teams_list, index=0)
home_team = c2.selectbox("Home Team", teams_list, index=min(1, len(teams_list)-1))

def goalie_list(team):
    g = GOALIES[GOALIES["Team"].astype(str) == str(team)]
    return sorted(g["Goalie"].dropna().astype(str).unique().tolist()) or ["(none)"]

c3, c4 = st.columns(2)
away_goalie = c3.selectbox("Away Goalie", goalie_list(away_team))
home_goalie = c4.selectbox("Home Goalie", goalie_list(home_team))

book_total = st.number_input("Book Total (optional)", min_value=0.0, max_value=20.0, step=0.5, value=6.0)

def pick_team_row(team):
    r = TEAMS[TEAMS["Team"].astype(str) == str(team)]
    rr = r.iloc[0]
    return dict(xGF_pg=float(rr["xGF_pg"]), xGA_pg=float(rr["xGA_pg"]),
                PP=(None if pd.isna(rr["PP%"]) else float(rr["PP%"])),
                PK=(None if pd.isna(rr["PK%"]) else float(rr["PK%"])))

def pick_goalie_row(team, goalie):
    g = GOALIES[(GOALIES["Team"].astype(str) == str(team)) & (GOALIES["Goalie"].astype(str) == str(goalie))]
    if g.empty: return dict(SV=None, GAA=None)
    gg = g.iloc[0]
    return dict(SV=(None if pd.isna(gg["SV%"]) else float(gg["SV%"])),
                GAA=(None if pd.isna(gg["GAA"]) else float(gg["GAA"])))

a = pick_team_row(away_team)
h = pick_team_row(home_team)
ag = pick_goalie_row(away_team, away_goalie)
hg = pick_goalie_row(home_team, home_goalie)

single = pd.DataFrame([{
    "Away Team": away_team, "Home Team": home_team,
    "Away xGF_pg": a["xGF_pg"], "Away xGA_pg": a["xGA_pg"],
    "Home xGF_pg": h["xGF_pg"], "Home xGA_pg": h["xGA_pg"],
    "Away PP%": a["PP"], "Away PK%": a["PK"],
    "Home PP%": h["PP"], "Home PK%": h["PK"],
    "Away SV%": ag["SV"], "Away GAA": ag["GAA"],
    "Home SV%": hg["SV"], "Home GAA": hg["GAA"],
    "Book Total": book_total
}])

PARAMS = dict(
    SIMS=SIMS, TEAM_OFF_W=TEAM_OFF_W, OPP_DEF_W=OPP_DEF_W,
    GOALIE_SCALE=GOALIE_SCALE, GOALIE_CAP=GOALIE_CAP,
    PACE_BONUS=PACE_BONUS, HFA_GOALS=HFA_GOALS,
    LEAGUE_PP=LEAGUE_PP, LEAGUE_PK=LEAGUE_PK, PP_W=PP_W, PK_W=PK_W,
    CALIBRATE=CALIBRATE, RANGE_MIN=RANGE_MIN, RANGE_MAX=RANGE_MAX,
    CLIP_LOW=CLIP_LOW, CLIP_HIGH=CLIP_HIGH,
    LEAGUE_SV=0.910, LEAGUE_GAA=3.00
)

if st.button("🔮 Predict Single Game"):
    preds = simulate_batch(single, PARAMS)
    row = preds.iloc[0]
    st.subheader("Game Summary")
    st.markdown(
        f"**{row['Away Team']} @ {row['Home Team']}**  \n"
        f"Projected: **{row['Away μ']} – {row['Home μ']}**  (Total: **{row['Total (Model)']}**)  \n"
        f"Winner: **{row['Predicted Winner']}**  \n"
        f"Win %: {row['Away Team']} **{row['Away Win %']}%**  |  {row['Home Team']} **{row['Home Win %']}%**"
    )
    st.dataframe(preds, use_container_width=True)
    st.download_button("⬇️ Download CSV", preds.to_csv(index=False).encode("utf-8"),
                       file_name="nhl_single_prediction.csv", mime="text/csv")

st.divider()
st.subheader("Batch Predictions (optional Games file)")
if GAMES is not None:
    # Join features from TEAMS; goalies left None in batch for simplicity
    def grab(team):
        r = TEAMS[TEAMS["Team"].astype(str) == str(team)]
        rr = r.iloc[0]
        return dict(xGF_pg=float(rr["xGF_pg"]), xGA_pg=float(rr["xGA_pg"]),
                    PP=(None if pd.isna(rr["PP%"]) else float(rr["PP%"])),
                    PK=(None if pd.isna(rr["PK%"]) else float(rr["PK%"])))
    feats = []
    for _, row in GAMES.iterrows():
        ateam, hteam = row["Away Team"], row["Home Team"]
        af, hf = grab(ateam), grab(hteam)
        feats.append({
            "Away Team": ateam, "Home Team": hteam,
            "Away xGF_pg": af["xGF_pg"], "Away xGA_pg": af["xGA_pg"],
            "Home xGF_pg": hf["xGF_pg"], "Home xGA_pg": hf["xGA_pg"],
            "Away PP%": af["PP"], "Away PK%": af["PK"],
            "Home PP%": hf["PP"], "Home PK%": hf["PK"],
            "Away SV%": None, "Away GAA": None,
            "Home SV%": None, "Home GAA": None,
            "Book Total": row.get("Book Total", None)
        })
    batch_df = pd.DataFrame(feats)
    preds = simulate_batch(batch_df, PARAMS)
    st.dataframe(preds, use_container_width=True)
    st.download_button("⬇️ Download Batch CSV", preds.to_csv(index=False).encode("utf-8"),
                       file_name="nhl_batch_predictions.csv", mime="text/csv")
else:
    st.info("Add a Games CSV/XLSX in the repo (or URL) to enable batch predictions.")
