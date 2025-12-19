import nflreadpy as nfl
import pandas as pd
from sklearn.linear_model import LinearRegression

# ----------------------------
# Load 2025 play-by-play & player stats
# ----------------------------
pbp_25 = nfl.load_pbp(2025).to_pandas()
player_25 = nfl.load_player_stats(2025).to_pandas()

# ----------------------------
# Merge passer info into play-by-play
# ----------------------------
pbp_with_pos = pbp_25.merge(
    player_25[['player_id', 'player_name', 'position']],
    left_on='passer_player_id',
    right_on='player_id',
    how='left'
)

# ----------------------------
# Filter only QB plays
# ----------------------------
qb_plays = pbp_with_pos[pbp_with_pos['position'] == 'QB'].copy()

# Compute rushing EPA per play
qb_plays['rush_epa'] = qb_plays['total_home_rush_epa']
qb_plays.loc[qb_plays['home_team'] != qb_plays['posteam'], 'rush_epa'] = qb_plays['total_away_rush_epa']

# ----------------------------
# Aggregate QB stats per season
# ----------------------------
epa_cols = ['qb_epa', 'air_epa', 'yac_epa', 'comp_air_epa', 'comp_yac_epa', 'rush_epa']
qb_epa_season = qb_plays.groupby(['player_id', 'player_name', 'season'])[epa_cols].sum().reset_index()

# Count dropbacks
qb_dropbacks = qb_plays.groupby(['player_id', 'player_name', 'season']).size().rename('dropbacks').reset_index()
qb_epa_season = qb_epa_season.merge(qb_dropbacks, on=['player_id','player_name','season'])

# Filter QBs with enough dropbacks
qb_epa_season = qb_epa_season[qb_epa_season['dropbacks'] > 1000]

# ----------------------------
# Compute per-dropback stats
# ----------------------------
qb_epa_season['epa_per_dropback'] = qb_epa_season['qb_epa'] / qb_epa_season['dropbacks']
qb_epa_season['air_epa_per_dropback'] = qb_epa_season['air_epa'] / qb_epa_season['dropbacks']
qb_epa_season['yac_epa_per_dropback'] = qb_epa_season['yac_epa'] / qb_epa_season['dropbacks']
qb_epa_season['rush_epa_per_dropback'] = qb_epa_season['rush_epa'] / qb_epa_season['dropbacks']

# ----------------------------
# Safe composite metric
# ----------------------------
epsilon = 0.01  # prevent division by near-zero
qb_epa_season['comp_adj_epa'] = (
    (qb_epa_season['comp_air_epa'] / qb_epa_season['air_epa'].replace(0, epsilon)) +
    (qb_epa_season['comp_yac_epa'] / qb_epa_season['yac_epa'].replace(0, epsilon))
) / 2
# Cap extreme outliers
qb_epa_season['comp_adj_epa'] = qb_epa_season['comp_adj_epa'].clip(-3, 3)

# ----------------------------
# Normalize for composite score
# ----------------------------
for col in ['qb_epa', 'epa_per_dropback', 'comp_adj_epa', 'rush_epa_per_dropback']:
    min_val = qb_epa_season[col].min()
    max_val = qb_epa_season[col].max()
    qb_epa_season[col + '_norm'] = (qb_epa_season[col] - min_val) / (max_val - min_val)

# Optional: cap rushing EPA to reduce extreme influence
qb_epa_season['rush_epa_per_dropback_capped'] = qb_epa_season['rush_epa_per_dropback'].clip(lower=-1.0, upper=1.5)
# Normalize capped rushing EPA
min_val = qb_epa_season['rush_epa_per_dropback_capped'].min()
max_val = qb_epa_season['rush_epa_per_dropback_capped'].max()
qb_epa_season['rush_epa_per_dropback_norm'] = (qb_epa_season['rush_epa_per_dropback_capped'] - min_val) / (max_val - min_val)

# ----------------------------
# Compute final QB score including rushing
# ----------------------------
# ----------------------------
# Compute two QB scores
# ----------------------------
# 1. Pure EPA-based score (your original weights)
qb_epa_season['qb_score_with_rush'] = (
    0.4 * qb_epa_season['qb_epa_norm'] +
    0.2 * qb_epa_season['epa_per_dropback_norm'] +
    0.2 * qb_epa_season['comp_adj_epa_norm'] +
    0.2 * qb_epa_season['rush_epa_per_dropback_norm']
)

# Map QB to main team
qb_teams = qb_plays.groupby(['player_id','player_name'])['posteam'].agg(lambda x: x.mode()[0]).reset_index()
qb_stats = qb_epa_season.merge(qb_teams, on=['player_id','player_name'], how='left')

# Compute team wins
pbp_sched_25 = nfl.load_schedules(2025).to_pandas()
def compute_wins(df, season):
    reg_games = df[(df['game_type'] == 'REG') & (df['season'] == season)]
    home_wins = reg_games.groupby('home_team')[['home_score','away_score']].apply(lambda x: (x['home_score'] > x['away_score']).sum())
    away_wins = reg_games.groupby('away_team')[['home_score','away_score']].apply(lambda x: (x['away_score'] > x['home_score']).sum())
    return pd.DataFrame({'team': home_wins.index, 'wins': home_wins.values + away_wins.values})

wins_25 = compute_wins(pbp_sched_25, 2025)

# Merge wins into QB stats
qb_stats = qb_stats.merge(wins_25, left_on='posteam', right_on='team', how='left')

# 2. Wins-based score using regression
X = qb_stats[['qb_epa_norm','epa_per_dropback_norm','comp_adj_epa_norm','rush_epa_per_dropback_norm']]
y = qb_stats['wins']

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Save regression weights as "qb_score_for_wins"
qb_stats['qb_score_for_wins'] = (
    model.coef_[0] * qb_stats['qb_epa_norm'] +
    model.coef_[1] * qb_stats['epa_per_dropback_norm'] +
    model.coef_[2] * qb_stats['comp_adj_epa_norm'] +
    model.coef_[3] * qb_stats['rush_epa_per_dropback_norm'] +
    model.intercept_
)

# Rank both scores
# Rank both scores
qb_stats['rank_qb_score_with_rush'] = qb_stats['qb_score_with_rush'].rank(ascending=False, method='min').astype(int)
qb_stats['rank_qb_score_for_wins'] = qb_stats['qb_score_for_wins'].rank(ascending=False, method='min').astype(int)
qb_stats['rank_total_epa'] = qb_stats['qb_epa'].rank(ascending=False, method='min').astype(int)

# Columns to display
display_cols = [
    'rank_qb_score_with_rush', 'rank_qb_score_for_wins', 'rank_total_epa',
    'player_name','season','qb_epa','rush_epa','epa_per_dropback','comp_adj_epa','rush_epa_per_dropback',
    'qb_score_with_rush','qb_score_for_wins'
]

# Sort and show top QBs
top_qbs = qb_stats.sort_values('qb_score_with_rush', ascending=False)
print(top_qbs[display_cols].head(20))

top_qbs = qb_stats.sort_values('qb_score_for_wins', ascending=False)
print(top_qbs[display_cols].head(20))
# Optional: check R^2 for wins regression
from sklearn.metrics import r2_score, mean_absolute_error
print("R^2 for wins regression:", r2_score(y, y_pred))
print("MAE:", mean_absolute_error(y, y_pred))
