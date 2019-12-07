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

college_trend = college_stats.groupby('college_pid')['college_points'].apply(lambda x: 
   x.diff().mean()).reset_index(name='avg_diff').fillna(0)

college_summary = college_stats.groupby("college_pid").agg(['mean', 'max']).reset_index()
college_summary.columns = college_summary.columns.map('_'.join).str.strip('_')
college_summary = college_summary.merge(college_trend, on='college_pid')

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


imp_columns = ['QB', 'RB', 'TE', 'WR', 'height', 'weight', 'bmi', 'arm_length', 
'hand_size', 'front_shoulder', 'back_shoulder', 'wonderlic', 'pass_velocity', 'ten_yard', 'twenty_yard',
'forty_yard', 'bench_press', 'vertical_leap', 'broad_jump', 'shuttle', 'sixty_shuttle', 'three_cone',
'four_square', 'games_mean', 'games_max', 'cmp_mean', 'cmp_max', 'pass_att_mean', 'pass_att_max', 
'pass_yards_mean', 'pass_yards_max', 'pass_td_mean', 'pass_td_max', 'intcp_mean', 'intcp_max',
'rating_mean', 'rating_max', 'rush_att_mean', 'rush_att_max',
'rush_yards_mean', 'rush_yards_max', 'rush_td_mean', 'rush_td_max', 'rec_mean', 'rec_max', 'rec_yards_mean',
'rec_yards_max', 'rec_td_mean', 'rec_td_max', 'college_points_mean', 'college_points_max', 'avg_diff', 'age']

imp_numeric = all_data[['QB', 'RB', 'TE', 'WR', 'height', 'weight', 'bmi', 'arm_length',
'hand_size', 'front_shoulder', 'back_shoulder', 'wonderlic', 'pass_velocity', 'ten_yard', 
'twenty_yard', 'forty_yard', 'bench_press', 'vertical_leap', 'broad_jump', 'shuttle', 'sixty_shuttle',
'three_cone', 'four_square', 'games_mean', 'games_max', 'cmp_mean', 'cmp_max', 'pass_att_mean', 'pass_att_max', 
'pass_yards_mean', 'pass_yards_max', 'pass_td_mean', 'pass_td_max', 'intcp_mean', 'intcp_max',
'rating_mean', 'rating_max', 'rush_att_mean', 'rush_att_max',
'rush_yards_mean', 'rush_yards_max', 'rush_td_mean', 'rush_td_max', 'rec_mean', 'rec_max', 'rec_yards_mean',
'rec_yards_max', 'rec_td_mean', 'rec_td_max', 'college_points_mean', 'college_points_max', 'avg_diff', 'age']].values

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

# summary_stats = stats.groupby("player_id").agg(['count', 'sum', 'mean', 'std', 'first', 'last', 'max', 'min', q25, 'median',
#                                 q75])['points'].reset_index()

summary_stats = stats.groupby("player_id").mean()['points'].reset_index()

master_data_clean = master_data.merge(summary_stats, on='player_id', how='inner')

qb_data = master_data_clean[(master_data_clean.position == "QB")].copy()


qb_data = qb_data[['player_id', 'points', 'college_pid', 'name', 'birth_year', 'birth_date', 'draft_year',
'age', 'round', 'pick', 'position', 'height', 'weight', 'bmi', 'arm_length', 'hand_size', 'front_shoulder', 
'back_shoulder', 'wonderlic', 'pass_velocity', 'ten_yard', 'twenty_yard', 'forty_yard', 'bench_press', 
'vertical_leap', 'broad_jump', 'shuttle', 'sixty_shuttle', 'three_cone', 'four_square', 'speed_score',
'agility_score', 'burst_score',  'height_adj_ss', 'games_mean', 'games_max', 'cmp_mean', 'cmp_max', 
'pass_att_mean', 'pass_att_max', 'pass_yards_mean', 'pass_yards_max', 'pass_td_mean', 'pass_td_max',
'intcp_mean', 'intcp_max', 'rating_mean', 'rating_max', 'rush_att_mean', 'rush_att_max', 
'rush_yards_mean', 'rush_yards_max', 'rush_td_mean', 'rush_td_max', 'college_points_mean',
 'college_points_max', 'avg_diff']]

#qb_data.corr()[['points']].transpose().to_csv('QBCorr.csv')
# qb_data.to_csv('QBPlayerData.csv', index=False)
# master_data.to_csv('NFLPlayerData.csv', 
#                 index=False)

qb_model = sm.formula.glm("points ~ (age + pick + height + weight \
    + forty_yard + cmp_max + pass_att_max + pass_yards_max + pass_td_max \
    + rush_att_max + rush_yards_max + rush_td_max)",
                        family=sm.families.Poisson(), data=qb_data).fit()
    
print(qb_model.summary())

print("Mean Squared Error: ", mse(qb_model.predict(qb_data), qb_data.points))

# 2019 Predictions... 
qb_19 = master_data[(master_data.draft_year == 2019) & (master_data.position == 'QB')].copy()

qb_19['predicted_points'] = qb_model.predict(qb_19)
qb_19 = qb_19[['name', 'predicted_points', 'draft_year',
'age', 'round', 'pick', 'height', 'weight', 'forty_yard', 'cmp_max', 'pass_att_max',
'pass_yards_max', 'pass_td_max', 'rush_att_max', 'rush_yards_max', 'rush_td_max']]

qb_19.to_csv('QBPred.csv', index=False)