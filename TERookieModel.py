# coding: utf-8

# packages needed
import pandas as pd
import numpy as np
from fancyimpute import KNN
import statsmodels.api as sm
from statsmodels.tools.eval_measures import mse
import os

file_path = os.path.dirname(os.path.abspath("__file__"))

df = pd.read_csv(file_path + "\\player_bio.csv", encoding="utf-8")

stats = pd.read_csv(file_path + "\\player_nfl_stats.csv", encoding="utf-8")

college_stats = pd.read_csv(file_path + "\\player_college_stats.csv", encoding="utf-8")

# applying season ranks by position and points
# stats['rank'] = stats.groupby(['season', 'position']).points.rank(method='dense', ascending=False)

print(df[(df.pick.isnull() == True) & (df.draft_year1.isnull() == False)])

college_stats['college_points'] = college_stats.pass_yards * 0.04 + college_stats.pass_td * 4 + \
(college_stats.intcp * -2) + ((college_stats.rush_yards + college_stats.rec_yards) * 0.10) + \
((college_stats.rush_td + college_stats.rec_td) * 6) + college_stats.rec * 0.5

# college_summary = college_stats.groupby("college_pid").agg(['count', 'sum', 'mean', 'first',
#                                                             'last', 'max', 'min', 'median']).reset_index()
college_trend = college_stats.groupby('college_pid')['college_points'].apply(lambda x: 
   x.diff().mean()).reset_index(name='avg_diff').fillna(0)


college_summary = college_stats.groupby("college_pid").last().reset_index()
college_summary = college_summary.merge(college_trend, on='college_pid')

# college_summary.columns = college_summary.columns.map('_'.join).str.strip('_')
df['draft_year'].fillna(df['draft_year1'], inplace=True)
df['position'].fillna(df['position1'], inplace=True)

all_data = df.merge(college_summary, on='college_pid', how='inner')

all_data['birth_year'] = [x[0:4] for x in all_data.birth_date.astype(str)]
all_data['age'] = all_data.draft_year - all_data.birth_year.astype(int)

# replace missing combine values with null for imputing
all_data.arm_length.replace(0, np.nan, inplace=True)
all_data.hand_size.replace(0, np.nan, inplace=True)
all_data.front_shoulder.replace(0, np.nan, inplace=True)
all_data.back_shoulder.replace(0, np.nan, inplace=True)
all_data.wonderlic.replace(0, np.nan, inplace=True)
all_data.pass_velocity.replace(0, np.nan, inplace=True)
all_data.ten_yard.replace(0, np.nan, inplace=True)
all_data.twenty_yard.replace(0, np.nan, inplace=True)
all_data.forty_yard.replace(0, np.nan, inplace=True)
all_data.bench_press.replace(0, np.nan, inplace=True)
all_data.vertical_leap.replace(0, np.nan, inplace=True)
all_data.broad_jump.replace(0, np.nan, inplace=True)
all_data.shuttle.replace(0, np.nan, inplace=True)
all_data.sixty_shuttle.replace(0, np.nan, inplace=True)
all_data.three_cone.replace(0, np.nan, inplace=True)
all_data.four_square.replace(0, np.nan, inplace=True)

#position dummy variables 
all_data[['QB', 'RB', 'TE', 'WR']] = pd.get_dummies(all_data['position'])


imp_columns = ['QB', 'RB', 'TE', 'WR', 'height', 'weight', 'bmi', 'arm_length', 'hand_size', 'front_shoulder', 'back_shoulder',
'wonderlic', 'pass_velocity', 'ten_yard', 'twenty_yard', 'forty_yard', 'bench_press', 'vertical_leap', 'broad_jump',
'shuttle', 'sixty_shuttle', 'three_cone', 'four_square', 'games', 'cmp', 'pass_att', 'pass_yards', 'pass_td', 'intcp',
'rating', 'rush_att', 'rush_yards', 'rush_td', 'rec', 'rec_yards', 'rec_td', 'college_points', 'avg_diff', 'age']

imp_numeric = all_data[['QB', 'RB', 'TE', 'WR', 'height', 'weight', 'bmi', 'arm_length', 'hand_size', 'front_shoulder',
'back_shoulder', 'wonderlic', 'pass_velocity', 'ten_yard', 'twenty_yard', 'forty_yard', 'bench_press', 'vertical_leap', 
'broad_jump', 'shuttle', 'sixty_shuttle', 'three_cone', 'four_square', 'games', 'cmp', 'pass_att', 'pass_yards', 'pass_td', 
'intcp', 'rating', 'rush_att', 'rush_yards', 'rush_td', 'rec', 'rec_yards', 'rec_td', 'college_points', 
                        'avg_diff', 'age']].values

# KNN imputing
imp = pd.DataFrame(KNN(k=5).fit_transform(imp_numeric), columns=imp_columns)

# add imputed values rest of dataset
all_data_imp = all_data.drop(imp_columns, axis=1)

master_data = all_data_imp.merge(imp, left_index=True, right_index=True)

# new combine variables
master_data['speed_score'] = (master_data.weight * 200)/(master_data.forty_yard**4)
master_data['agility_score'] = master_data.three_cone + master_data.shuttle
master_data['height_adj_ss'] = master_data.speed_score * (master_data.height / 73.5) ** 1.5
master_data['burst_score'] = master_data.vertical_leap + master_data.broad_jump
# catch radius and weight adjusted bench ?

# merge and drop players without combine data or without any stats
stats = stats.merge(df[['player_id', 'draft_year']], on='player_id', how='outer')

# remove all seasons except first four
stats = stats[(stats.season - stats.draft_year) < 4]

summary_stats = stats.groupby("player_id").mean()['points'].reset_index()

master_data_clean = master_data.merge(summary_stats, on='player_id', how='inner')


te_data = master_data_clean[master_data_clean.position == 'TE']

te_data = te_data[['player_id', 'points', 'college_pid', 'name', 'birth_year', 'birth_date', 'draft_year', 'age', 'round', 
'pick', 'position', 'height', 'weight', 'bmi', 'arm_length', 'hand_size', 'front_shoulder', 'back_shoulder', 'wonderlic',
'ten_yard', 'twenty_yard', 'forty_yard', 'bench_press', 'vertical_leap', 'broad_jump', 'shuttle', 'sixty_shuttle', 'three_cone',
'four_square', 'speed_score', 'agility_score', 'burst_score',  'height_adj_ss', 'games', 'rec', 'rec_yards', 'rec_td',
                   'college_points', 'avg_diff']]

#te_data.corr()[['points']].transpose().to_csv('TECorr.csv')
# te_data.to_csv('TEPlayerData.csv')
# master_data.to_csv('NFLPlayerData.csv', 
#                 index=False)

te_model = sm.formula.glm("points ~ (pick + arm_length + ten_yard \
+ forty_yard + burst_score \
+ rec + rec_yards + rec_td)", family=sm.families.Poisson(), data=te_data).fit()

print(te_model.summary())

print("Mean Squared Error: ", mse(te_model.predict(te_data), te_data.points))

# 2019 Predictions... TJ Hockenson & Noah Fant low? Trevon Wisco? Depth Chart?
te_19 = master_data[(master_data.draft_year == 2019) & (master_data.position == 'TE')].copy()
te_19['predicted_points'] = te_model.predict(te_19)

te_19 = te_19[['name', 'predicted_points', 'draft_year',
'age', 'round', 'pick', 'height', 'arm_length', 'ten_yard', 'twenty_yard',
'forty_yard', 'burst_score', 'rec', 'rec_yards', 'rec_td']]

te_19.to_csv('TEPred.csv', index=False)