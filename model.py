from sklearn.ensemble import RandomForestRegressor
import pandas as pd

y_train = train.pop('target') # train is your submission!
rf = RandomForestRegressor(
       n_estimators=1000,
       max_depth=7,
       n_jobs=-1,
       random_state=42)
rf.fit(train, y_train)
y_hat = rf.predict(test)