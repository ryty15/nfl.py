import nflreadpy as nfl
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ----------------------------
# Load 2024 play-by-play & schedules
# ----------------------------
pbp_24 = nfl.load_pbp(2024).to_pandas()
schedules_24 = nfl.load_schedules().to_pandas()

def compute_wins(df, season):
    reg_games = df[(df['game_type'] == 'REG') & (df['season'] == season)]
    home_wins = reg_games.groupby('home_team').apply(lambda x: (x['home_score'] > x['away_score']).sum())
    away_wins = reg_games.groupby('away_team').apply(lambda x: (x['away_score'] > x['home_score']).sum())
    return pd.DataFrame({'team': home_wins.index, 'wins': home_wins.values + away_wins.values})

def compute_team_epa_per_game(pbp):
    pbp_reg = pbp[(pbp['season_type'] == 'REG') & (pbp['epa'].notna())]
    
    # Total EPA per team
    off_epa = (
        pbp_reg.groupby(['season', 'posteam'])['epa']
        .sum()
        .reset_index()
        .rename(columns={'posteam':'team', 'epa':'off_epa'})
    )
    def_epa = (
        pbp_reg.groupby(['season', 'defteam'])['epa']
        .sum()
        .reset_index()
        .rename(columns={'defteam':'team', 'epa':'def_epa_allowed'})
    )
    
    # Games played per team
    games_off = pbp_reg.groupby('posteam')['game_id'].nunique().reset_index().rename(columns={'posteam':'team','game_id':'games_played'})
    games_def = pbp_reg.groupby('defteam')['game_id'].nunique().reset_index().rename(columns={'defteam':'team','game_id':'games_played'})
    
    # Merge games played
    off_epa = off_epa.merge(games_off, on='team')
    def_epa = def_epa.merge(games_def, on='team')
    
    # Compute per-game EPA
    off_epa['off_epa_per_game'] = off_epa['off_epa'] / off_epa['games_played']
    off_epa = off_epa.rename(columns={'games_played':'games_played_off'})
    
    def_epa['def_epa_allowed_per_game'] = def_epa['def_epa_allowed'] / def_epa['games_played']
    def_epa = def_epa.rename(columns={'games_played':'games_played_def'})
    
    # Merge offense and defense
    team_epa = off_epa.merge(def_epa, on=['season','team'])
    team_epa['net_epa_per_game'] = team_epa['off_epa_per_game'] - team_epa['def_epa_allowed_per_game']
    
    return team_epa[['team','off_epa_per_game','def_epa_allowed_per_game','net_epa_per_game','games_played_off','games_played_def']]



# ----------------------------
# Train model on 2024
# ----------------------------
wins_24 = compute_wins(schedules_24, 2024)
team_epa_24 = compute_team_epa_per_game(pbp_24)

X_24 = team_epa_24[['off_epa_per_game', 'def_epa_allowed_per_game']]
y_24 = team_epa_24.merge(wins_24, on='team')['wins']

model = LinearRegression()
model.fit(X_24, y_24)

# ----------------------------
# Load 2025 play-by-play & schedules
# ----------------------------
pbp_25 = nfl.load_pbp(2025).to_pandas()
schedules_25 = nfl.load_schedules().to_pandas()

team_epa_25 = compute_team_epa_per_game(pbp_25)
wins_so_far_25 = compute_wins(schedules_25, 2025)

# Merge wins so far
team_epa_25 = team_epa_25.merge(wins_so_far_25, on='team', how='left')
team_epa_25['wins_so_far'] = team_epa_25['wins'].fillna(0)

# ----------------------------
# Predict remaining wins (games 15-17)
# ----------------------------
remaining_games = 3
total_games = 17

# Predicted total wins for full season (using EPA per game * 17)
team_epa_25['predicted_total_wins'] = model.predict(team_epa_25[['off_epa_per_game','def_epa_allowed_per_game']])

# Convert to predicted wins per game
team_epa_25['predicted_wins_per_game'] = team_epa_25['predicted_total_wins'] / total_games

# Predicted wins for remaining 3 games
team_epa_25['predicted_remaining_wins'] = team_epa_25['predicted_wins_per_game'] * remaining_games

# Predicted total wins for full season (using EPA per game * 17)
team_epa_25['predicted_total_wins_upd'] = team_epa_25['wins_so_far']+team_epa_25['predicted_remaining_wins']
# ----------------------------
# Show results

# ----------------------------
# Show results
# ----------------------------
print(team_epa_25[['team', 
                   'wins_so_far', 
                   'predicted_remaining_wins', 
                   'predicted_total_wins', 'predicted_total_wins_upd']  # <-- include total predicted wins
      ].sort_values('predicted_total_wins_upd', ascending=False))
