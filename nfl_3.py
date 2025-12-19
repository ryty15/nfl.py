import nflreadpy as nfl
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ----------------------------
# Load play-by-play data
# ----------------------------
pbp_24 = nfl.load_pbp(2025).to_pandas()

# ----------------------------
# Load schedules (for wins)
# ----------------------------
schedules = nfl.load_schedules().to_pandas()

def compute_wins(df, season):
    """Compute total regular season wins for each team in a given season"""
    reg_games = df[(df['game_type'] == 'REG') & (df['season'] == season)]
    
    home_wins = reg_games.groupby('home_team').apply(
        lambda x: (x['home_score'] > x['away_score']).sum()
    )
    away_wins = reg_games.groupby('away_team').apply(
        lambda x: (x['away_score'] > x['home_score']).sum()
    )
    
    wins_df = pd.DataFrame({
        'team': home_wins.index,
        'wins': home_wins.values + away_wins.values
    })
    return wins_df

def compute_team_epa(pbp):
    """Compute offensive, defensive, and net EPA per team per season"""
    pbp_reg = pbp[(pbp['season_type'] == 'REG') & (pbp['epa'].notna())]

    off_epa = (
        pbp_reg.groupby(['season', 'posteam'])['epa']
        .sum()
        .reset_index()
        .rename(columns={'posteam':'team','epa':'off_epa'})
    )

    def_epa = (
        pbp_reg.groupby(['season', 'defteam'])['epa']
        .sum()
        .reset_index()
        .rename(columns={'defteam':'team','epa':'def_epa_allowed'})
    )

    team_epa = off_epa.merge(def_epa, on=['season','team'])
    team_epa['net_epa'] = team_epa['off_epa'] - team_epa['def_epa_allowed']
    return team_epa

# ----------------------------
# Compute 2024 wins and EPA
# ----------------------------
wins_24 = compute_wins(schedules, 2025)
team_epa_24 = compute_team_epa(pbp_24)

# Merge EPA with wins
team_epa_24 = team_epa_24.merge(wins_24, on='team', how='left')

# ----------------------------
# Fit linear regression within 2024
# ----------------------------
X = team_epa_24[['off_epa', 'def_epa_allowed']]
y = team_epa_24['wins']

model = LinearRegression()
model.fit(X, y)

# Predict wins
team_epa_24['predicted_wins'] = model.predict(X)

# ----------------------------
# Evaluate predictions
# ----------------------------
print("Intercept:", model.intercept_)
print("Coefficients:")
print(pd.Series(model.coef_, index=X.columns))

print("MAE:", mean_absolute_error(y, team_epa_24['predicted_wins']))
print("R²:", r2_score(y, team_epa_24['predicted_wins']))

# ----------------------------
# Plot actual vs predicted wins
# ----------------------------
plt.figure(figsize=(6,6))
plt.scatter(team_epa_24['wins'], team_epa_24['predicted_wins'])
plt.plot([0, 17], [0, 17], linestyle='--', color='gray')
plt.xlabel('Actual Wins (2024)')
plt.ylabel('Predicted Wins')
plt.title('Predicted vs Actual Wins (2024)')
plt.tight_layout()
plt.show()

# ----------------------------
# Show predicted vs actual
# ----------------------------
print(team_epa_24[['team', 'wins', 'predicted_wins']].sort_values('wins', ascending=False))

#findings, win-predicteed= 0.6-0.7 -> super bowl winner
# winning one game likely if win-predicted= 1.5
#KC outlier made super bowl final with 4.7 difference, MIN lost to LA (3 vs 1.5
#theory: if 3 vs 1.5 first round winner= 1.5) if 3 vs 1.5 second round 3 wins
#or maybe need above 4-4.5 to be true outlier and beat a 1.5 team and go to super bowl

