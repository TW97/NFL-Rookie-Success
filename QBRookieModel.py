# coding: utf-8
# packages needed
import pandas as pd
import numpy as np
from fancyimpute import KNN
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Lasso, ElasticNet, ElasticNetCV, Ridge, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, GradientBoostingRegressor
import os

file_path = os.path.dirname(os.path.abspath("__file__"))

df = pd.read_csv(file_path + "\\player_bio.csv", encoding="utf-8")

stats = pd.read_csv(file_path + "\\player_nfl_stats.csv", encoding="utf-8")

college_stats = pd.read_csv(file_path + "\\player_college_stats.csv", encoding="utf-8")

# applying season ranks by position and points
# stats['rank'] = stats.groupby(['season', 'position']).points.rank(method='dense', ascending=False)

def q25(x):
    return x.quantile(0.25)

def q75(x):
    return x.quantile(0.75)

print(df[(df.pick.isnull() == True) & (df.draft_year1.isnull() == False)])

college_stats['college_points'] = college_stats.pass_yards * 0.04 + college_stats.pass_td * 4 + \
(college_stats.intcp * -2) + ((college_stats.rush_yards + college_stats.rec_yards) * 0.10) + \
((college_stats.rush_td + college_stats.rec_td) * 6) + college_stats.rec * 0.5

# college_summary = college_stats.groupby("college_pid").agg(['count', 'sum', 'mean', 'first',
#                                                             'last', 'max', 'min', 'median']).reset_index()
college_trend = college_stats.groupby('college_pid')['college_points'].apply(lambda x: 
   x.diff().mean()).reset_index(name='avg_diff').fillna(0)

college_summary = college_stats.groupby("college_pid").agg(['mean', 'max']).reset_index()
college_summary.columns = college_summary.columns.map('_'.join).str.strip('_')
college_summary = college_summary.merge(college_trend, on='college_pid')

df['draft_year'].fillna(df['draft_year1'], inplace=True)
df['position'].fillna(df['position1'], inplace=True)

all_data = df.merge(college_summary, on='college_pid', how='inner')

all_data = all_data[all_data.name != "Dan Vitale"]

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


imp_columns = ['QB', 'RB', 'TE', 'WR', 'round', 'pick', 'height', 'weight', 'bmi', 'arm_length', 
'hand_size', 'front_shoulder', 'back_shoulder', 'wonderlic', 'pass_velocity', 'ten_yard', 'twenty_yard',
'forty_yard', 'bench_press', 'vertical_leap', 'broad_jump', 'shuttle', 'sixty_shuttle', 'three_cone',
'four_square', 'games_mean', 'games_max', 'cmp_mean', 'cmp_max', 'pass_att_mean', 'pass_att_max', 
'pass_yards_mean', 'pass_yards_max', 'pass_td_mean', 'pass_td_max', 'intcp_mean', 'intcp_max',
'rating_mean', 'rating_max', 'rush_att_mean', 'rush_att_max',
'rush_yards_mean', 'rush_yards_max', 'rush_td_mean', 'rush_td_max', 'rec_mean', 'rec_max', 'rec_yards_mean',
'rec_yards_max', 'rec_td_mean', 'rec_td_max', 'college_points_mean', 'college_points_max', 'avg_diff', 'age']

imp_numeric = all_data[['QB', 'RB', 'TE', 'WR', 'round', 'pick', 'height', 'weight', 'bmi', 'arm_length',
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

qb_data = master_data_clean[master_data_clean.position == 'QB']

# 'count', 'sum', 'mean', 'std', 'first', 'last', 'max', 'min', 'median'

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


X = qb_data[['age', 'pick', 'bench_press', 'pass_velocity', 'broad_jump', 'agility_score',
             'height_adj_ss', 'college_points_max', 'avg_diff']]

y = qb_data['points']


alphas = [0.0001, 0.001, 0.01, 0.3, 0.5, 0.7, 1]

for a in alphas:
    model = ElasticNet(alpha=a).fit(X, y)
    pred_y = model.predict(X)
    r2 = cross_val_score(model, X, y, cv=5, scoring="r2")
    mse = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
    print("Alpha:{0:.4f}, R2 Max:{1:.2f}, R2 Avg:{2:.2f}, R2 Min:{3:.2f}, R2 Stdev:{4:.2f}, \
          MSE Max:{5:.2f}, MSE Avg:{6:.2f}, MSE Min:{7:.2f}, MSE Stdev:{8:.2f}"
       .format(a, r2.max(), r2.mean(), r2.min(), r2.std(), mse.max(), mse.mean(), mse.min(), mse.std()))

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)
rd = Ridge()
ls = Lasso()
model = ElasticNet(alpha=0.3)
lr = LinearRegression()
#knr = KNeighborsRegressor()
#sv = SVR()
#dt = DecisionTreeRegressor()
#rfr = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=100)
#gbr = GradientBoostingRegressor(random_state=100, n_estimators=500)
#model = VotingRegressor(estimators=[('en', en), ('ls', ls)])
seasons = [x for x in range(1987, 2019)]
scores = []
for season in seasons:
    X_train = qb_data[qb_data.draft_year != season][['age', 'pick', 'bench_press',
            'pass_velocity', 'broad_jump', 'agility_score', 'height_adj_ss',
            'college_points_max', 'avg_diff']]
    X_test = qb_data[qb_data.draft_year == season][['age', 'pick', 'bench_press',
            'pass_velocity', 'broad_jump', 'agility_score', 'height_adj_ss',
            'college_points_max', 'avg_diff']]
    y_train = qb_data[qb_data.draft_year != season]['points']
    y_test = qb_data[qb_data.draft_year == season]['points']
    model = ElasticNet(alpha=0.3)
    model.fit(X_train, y_train)
    fit_score = model.score(X_train, y_train)
    predicted = model.predict(X_test)
    score = r2_score(y_test, predicted)
    print("Season: ", season, "\t", "Train R-Squared", fit_score, "\t", "Test R-Squared: ", score)

# 2019 Predictions... tweak the college points, adjust for running QB v Pocket Passer, add in conference?
qb_19 = master_data[(master_data.draft_year == 2019) & (master_data.position == 'QB')]
qb_19_x = qb_19[['age', 'pick', 'bench_press',
            'pass_velocity', 'broad_jump', 'agility_score', 'height_adj_ss', 
            'college_points_max', 'avg_diff']]

qb_model = ElasticNet(alpha=0.3)
qb_model.fit(X, y)
qb_predicted = qb_model.predict(qb_19_x)
print(pd.DataFrame(qb_model.coef_, qb_19_x.columns, columns=['Coefficient'])) 
print("Model Intercept: ", qb_model.intercept_) 
qb_19['predicted_points'] = qb_predicted
qb_19 = qb_19[['name', 'predicted_points', 'draft_year',
'age', 'pick', 'pass_velocity', 'bench_press', 'broad_jump', 'agility_score',
 'height_adj_ss','college_pid', 'avg_diff', 'college_points_max', 'player_id']]

qb_19.to_csv('QBPred.csv', index=False)