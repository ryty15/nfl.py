import nflreadpy as nfl
import matplotlib.pyplot as plt


# Load current season play-by-play data
pbp = nfl.load_pbp()
print(pbp.head())
pbp = pbp.to_pandas()

team_stats = nfl.load_team_stats(seasons=True)



import pandas as pd

#2025
pbp_25= nfl.load_pbp(2025)
pbp_25=pbp_25.to_pandas()

#2024
pbp_24= nfl.load_pbp(2024)
pbp_24=pbp_24.to_pandas()

#2023
pbp_23= nfl.load_pbp(2023)
pbp_23=pbp_23.to_pandas()



# Only regular season plays
pbp_reg = pbp_25[
    (pbp_25['season_type'] == 'REG') &
    (pbp_25['epa'].notna())
]


# Offensive EPA per team per season
off_epa = (
    pbp_reg
    .groupby(['season', 'posteam'])['epa']
    .sum()
    .reset_index()
    .rename(columns={'posteam': 'team', 'epa': 'off_epa'})
)

# Defensive EPA allowed per team per season
def_epa = (
    pbp_reg
    .groupby(['season', 'defteam'])['epa']
    .sum()
    .reset_index()
    .rename(columns={'defteam': 'team', 'epa': 'def_epa_allowed'})
)

# Merge
team_epa = off_epa.merge(def_epa, on=['season', 'team'])

# Net EPA
team_epa['net_epa'] = team_epa['off_epa'] - team_epa['def_epa_allowed']
print(team_epa.sort_values('net_epa', ascending=False).head())
print(pbp_25['epa'].describe())
team_epa_sorted = team_epa.sort_values('net_epa')

#2025
plt.figure(figsize=(8, 8))
plt.scatter(team_epa['off_epa'], -team_epa['def_epa_allowed'])

for _, row in team_epa.iterrows():
    plt.text(row['off_epa'], -row['def_epa_allowed'], row['team'], fontsize=8)

plt.axhline(0, linestyle='--')
plt.axvline(0, linestyle='--')
plt.xlabel('Offensive EPA')
plt.ylabel('Defensive EPA (positive = good)')
plt.title('2025 NFL Team EPA Profile')
plt.tight_layout()
plt.savefig("epa_scatter_2025.png", dpi=300)
plt.close()

#2024
pbp_reg = pbp_24[
    (pbp_24['season_type'] == 'REG') &
    (pbp_24['epa'].notna())
]


# Offensive EPA per team per season
off_epa = (
    pbp_reg
    .groupby(['season', 'posteam'])['epa']
    .sum()
    .reset_index()
    .rename(columns={'posteam': 'team', 'epa': 'off_epa'})
)

# Defensive EPA allowed per team per season
def_epa = (
    pbp_reg
    .groupby(['season', 'defteam'])['epa']
    .sum()
    .reset_index()
    .rename(columns={'defteam': 'team', 'epa': 'def_epa_allowed'})
)
# Net EPA
team_epa['net_epa'] = team_epa['off_epa'] - team_epa['def_epa_allowed']
print(team_epa.sort_values('net_epa', ascending=False).head())
print(pbp_24['epa'].describe())
team_epa_sorted = team_epa.sort_values('net_epa')

# Merge
team_epa = off_epa.merge(def_epa, on=['season', 'team'])

#2024
plt.figure(figsize=(8, 8))
plt.scatter(team_epa['off_epa'], -team_epa['def_epa_allowed'])

for _, row in team_epa.iterrows():
    plt.text(row['off_epa'], -row['def_epa_allowed'], row['team'], fontsize=8)

plt.axhline(0, linestyle='--')
plt.axvline(0, linestyle='--')
plt.xlabel('Offensive EPA')
plt.ylabel('Defensive EPA (positive = good)')
plt.title('2024 NFL Team EPA Profile')
plt.tight_layout()
plt.savefig("epa_scatter_2024.png", dpi=300)
plt.close()




#2023
pbp_reg = pbp_23[
    (pbp_23['season_type'] == 'REG') &
    (pbp_23['epa'].notna())
]


# Offensive EPA per team per season
off_epa = (
    pbp_reg
    .groupby(['season', 'posteam'])['epa']
    .sum()
    .reset_index()
    .rename(columns={'posteam': 'team', 'epa': 'off_epa'})
)

# Defensive EPA allowed per team per season
def_epa = (
    pbp_reg
    .groupby(['season', 'defteam'])['epa']
    .sum()
    .reset_index()
    .rename(columns={'defteam': 'team', 'epa': 'def_epa_allowed'})
)
# Net EPA
team_epa['net_epa'] = team_epa['off_epa'] - team_epa['def_epa_allowed']
print(team_epa.sort_values('net_epa', ascending=False).head())
print(pbp_23['epa'].describe())
team_epa_sorted = team_epa.sort_values('net_epa')

# Merge
team_epa = off_epa.merge(def_epa, on=['season', 'team'])

#2024
plt.figure(figsize=(8, 8))
plt.scatter(team_epa['off_epa'], -team_epa['def_epa_allowed'])

for _, row in team_epa.iterrows():
    plt.text(row['off_epa'], -row['def_epa_allowed'], row['team'], fontsize=8)

plt.axhline(0, linestyle='--')
plt.axvline(0, linestyle='--')
plt.xlabel('Offensive EPA')
plt.ylabel('Defensive EPA (positive = good)')
plt.title('2023 NFL Team EPA Profile')
plt.tight_layout()
plt.savefig("epa_scatter_2023.png", dpi=300)
plt.close()


