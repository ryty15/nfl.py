import nflreadpy as nfl
import pandas as pd
import numpy as np
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


games_24 = schedules_24[
    (schedules_24['season'] == 2024) &
    (schedules_24['game_type'] == 'REG')
].copy()

games_24['point_diff'] = games_24['home_score'] - games_24['away_score']

epa = team_epa_24[['team','off_epa_per_game','def_epa_allowed_per_game']]

games_24 = games_24.merge(
    epa, left_on='home_team', right_on='team'
).rename(columns={
    'off_epa_per_game':'home_off_epa',
    'def_epa_allowed_per_game':'home_def_epa'
}).drop(columns='team')

games_24 = games_24.merge(
    epa, left_on='away_team', right_on='team'
).rename(columns={
    'off_epa_per_game':'away_off_epa',
    'def_epa_allowed_per_game':'away_def_epa'
}).drop(columns='team')

X = games_24[['home_off_epa','home_def_epa','away_off_epa','away_def_epa']]
y = games_24['point_diff']

game_model = LinearRegression()
game_model.fit(X, y)

import scipy.stats as stats

def win_prob_from_margin(mu, sigma=13.5):
    return 1 - stats.norm.cdf(0, loc=mu, scale=sigma)

HFA = 2.1  # points

def simulate_game(home_team, away_team, team_epa, model, n_sims=10000):
    home = team_epa.loc[team_epa.team == home_team].iloc[0]
    away = team_epa.loc[team_epa.team == away_team].iloc[0]

    X = pd.DataFrame([{
        'home_off_epa': home.off_epa_per_game,
        'home_def_epa': home.def_epa_allowed_per_game,
        'away_off_epa': away.off_epa_per_game,
        'away_def_epa': away.def_epa_allowed_per_game
    }])

    # Base margin from EPA
    mu = model.predict(X)[0]

    # 🔥 add home-field advantage
    mu += HFA

    sims = np.random.normal(mu, 13.5, n_sims)

    return {
        'expected_margin': mu,
        'home_win_prob': (sims > 0).mean(),
        'median_margin': np.median(sims),
        'p10_margin': np.percentile(sims, 10),
        'p90_margin': np.percentile(sims, 90)
    }


result = simulate_game(
    home_team='NYG',
    away_team='MIN',
    team_epa=team_epa_25,
    model=game_model,
    n_sims=1000000
)

print(result)









HFA = 2.1  # points

def simulate_game(home_team, away_team, team_epa, model, n_sims=10000):
    home = team_epa.loc[team_epa.team == home_team].iloc[0]
    away = team_epa.loc[team_epa.team == away_team].iloc[0]

    X = pd.DataFrame([{
        'home_off_epa': home.off_epa_per_game,
        'home_def_epa': home.def_epa_allowed_per_game,
        'away_off_epa': away.off_epa_per_game,
        'away_def_epa': away.def_epa_allowed_per_game
    }])

    # Base margin from EPA
    mu = model.predict(X)[0]

    # 🔥 add home-field advantage
    mu += HFA

    sims = np.random.normal(mu, 13.5, n_sims)

    return {
        'expected_margin': mu,
        'home_win_prob': (sims > 0).mean(),
        'median_margin': np.median(sims),
        'p10_margin': np.percentile(sims, 10),
        'p90_margin': np.percentile(sims, 90)
    }


remaining_games_25 = schedules_25[
    (schedules_25['season'] == 2025) &
    (schedules_25['game_type'] == 'REG') &
    (schedules_25['home_score'].isna())
][['week','home_team','away_team']]

def game_win_probs(games, team_epa, model):
    rows = []

    for _, g in games.iterrows():
        result = simulate_game(
            home_team=g.home_team,
            away_team=g.away_team,
            team_epa=team_epa,
            model=model,
            n_sims=20000
        )

        rows.append({
            'week': g.week,
            'home_team': g.home_team,
            'away_team': g.away_team,
            'home_win_prob': result['home_win_prob'],
            'away_win_prob': 1 - result['home_win_prob']
        })

    return pd.DataFrame(rows)
game_probs_25 = game_win_probs(
    remaining_games_25,
    team_epa_25,
    game_model
)

home_wins = (
    game_probs_25
    .groupby('home_team')['home_win_prob']
    .sum()
    .reset_index()
    .rename(columns={'home_team':'team', 'home_win_prob':'exp_wins'})
)

away_wins = (
    game_probs_25
    .groupby('away_team')['away_win_prob']
    .sum()
    .reset_index()
    .rename(columns={'away_team':'team', 'away_win_prob':'exp_wins'})
)

expected_remaining = (
    pd.concat([home_wins, away_wins])
    .groupby('team')['exp_wins']
    .sum()
    .reset_index()
)

final_projection = (
    team_epa_25[['team','wins_so_far']]
    .merge(expected_remaining, on='team', how='left')
    .fillna({'exp_wins': 0})
)

final_projection['projected_wins'] = (
    final_projection['wins_so_far'] +
    final_projection['exp_wins']
)

print(final_projection.sort_values('projected_wins', ascending=False))





#Monte Carlo
remaining_games_25 = schedules_25[
    (schedules_25['season'] == 2025) &
    (schedules_25['game_type'] == 'REG') &
    (schedules_25['home_score'].isna())
][['week','home_team','away_team']]


game_probs = []

for _, g in remaining_games_25.iterrows():
    result = simulate_game(
        home_team=g.home_team,
        away_team=g.away_team,
        team_epa=team_epa_25,
        model=game_model,
        n_sims=20000
    )

    game_probs.append({
        'week': g.week,
        'home_team': g.home_team,
        'away_team': g.away_team,
        'home_win_prob': result['home_win_prob']
    })

game_probs = pd.DataFrame(game_probs)


def simulate_seasons(game_probs, wins_so_far, n_sims=10000):
    teams = wins_so_far['team'].values
    win_matrix = pd.DataFrame(0, index=np.arange(n_sims), columns=teams)

    # Start with current wins
    for team in teams:
        win_matrix[team] = wins_so_far.loc[
            wins_so_far.team == team, 'wins_so_far'
        ].values[0]

    # Simulate remaining games
    for _, g in game_probs.iterrows():
        home = g.home_team
        away = g.away_team
        p = g.home_win_prob

        home_wins = np.random.binomial(1, p, n_sims)

        win_matrix[home] += home_wins
        win_matrix[away] += (1 - home_wins)

    return win_matrix

team_info = pd.DataFrame({
    'team': [
        'BUF','MIA','NE','NYJ',
        'BAL','CIN','CLE','PIT',
        'HOU','IND','JAX','TEN',
        'DEN','KC','LV','LAC',
        'DAL','NYG','PHI','WAS',
        'CHI','DET','GB','MIN',
        'ATL','CAR','NO','TB',
        'ARI','LA','SF','SEA'
    ],
    'conference': [
        'AFC','AFC','AFC','AFC',
        'AFC','AFC','AFC','AFC',
        'AFC','AFC','AFC','AFC',
        'AFC','AFC','AFC','AFC',
        'NFC','NFC','NFC','NFC',
        'NFC','NFC','NFC','NFC',
        'NFC','NFC','NFC','NFC',
        'NFC','NFC','NFC','NFC'
    ],
    'division': [
        'AFCE','AFCE','AFCE','AFCE',
        'AFCN','AFCN','AFCN','AFCN',
        'AFCS','AFCS','AFCS','AFCS',
        'AFCW','AFCW','AFCW','AFCW',
        'NFCE','NFCE','NFCE','NFCE',
        'NFCN','NFCN','NFCN','NFCN',
        'NFCS','NFCS','NFCS','NFCS',
        'NFCW','NFCW','NFCW','NFCW'
    ]
})
def get_playoff_teams(wins_row, team_info):
    df = pd.DataFrame({
        'team': wins_row.index,
        'wins': wins_row.values
    }).merge(team_info, on='team')

    playoff_teams = []

    for conf in ['AFC', 'NFC']:
        # subset to conference
        conf_df = df[df.conference == conf]

        # 🔀 randomize order to break ties fairly
        conf_df = conf_df.sample(frac=1)

        # 1️⃣ Division winners
        division_winners = (
            conf_df
            .sort_values('wins', ascending=False)
            .groupby('division')
            .head(1)
        )

        playoff_teams.extend(division_winners.team.tolist())

        # 2️⃣ Wild cards (next best 3)
        remaining = conf_df[
            ~conf_df.team.isin(division_winners.team)
        ]

        wild_cards = (
            remaining
            .sort_values('wins', ascending=False)
            .head(3)
        )

        playoff_teams.extend(wild_cards.team.tolist())

    return playoff_teams


def playoff_odds(season_sims, team_info):
    playoff_counts = pd.Series(0, index=season_sims.columns)

    for i in range(len(season_sims)):
        winners = get_playoff_teams(season_sims.iloc[i], team_info)
        playoff_counts[winners] += 1

    return (playoff_counts / len(season_sims)).sort_values(ascending=False)


wins_so_far = team_epa_25[['team','wins_so_far']]
season_sims = simulate_seasons(game_probs, wins_so_far, n_sims=20000)

expected_wins = season_sims.mean().sort_values(ascending=False)
print(expected_wins)
print(season_sims['BAL'].value_counts(normalize=True).sort_index())


# Get overall playoff odds
odds = playoff_odds(season_sims, team_info)

# Merge conference info
odds_df = odds.reset_index().rename(columns={'index':'team', 0:'playoff_odds'})
odds_df = odds_df.merge(team_info[['team','conference']], on='team')

# Split by conference
odds_afc = odds_df[odds_df.conference == 'AFC'].sort_values('playoff_odds', ascending=False)
odds_nfc = odds_df[odds_df.conference == 'NFC'].sort_values('playoff_odds', ascending=False)

print("AFC Playoff Odds:\n", odds_afc)
print("\nNFC Playoff Odds:\n", odds_nfc)

