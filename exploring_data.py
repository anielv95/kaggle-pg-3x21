import pandas as pd
import zipfile
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

with zipfile.ZipFile("/gh/kaggle-pg-3x21/data/playground-series-s3e21.zip") as z:
    # open the csv file in the dataset
    with z.open("sample_submission.csv") as f:
        # read the dataset
        train = pd.read_csv(f)
train.to_csv("/gh/kaggle-pg-3x21/kaggle-pg-3x21/data/sample.csv", index=False)

# kaggle competitions submit -c playground-series-s3e21 -f ./data/sample.csv -m "Message"

train = pd.read_csv("/gh/kaggle-pg-3x21/kaggle-pg-3x21/data/sample.csv")
"""train.head(2)
train.shape
train.info()

train.describe()"""

##############################
##############################
##############################
# this section was extracted from Aurélien Géron's repository
# https://github.com/ageron/handson-ml3/blob/main/02_end_to_end_machine_learning_project.ipynb
# He's the author of the great book:
# Hands-On Machine Learning with Scikit-Learn, Keras, and Tensorflow
# https://www.amazon.com.mx/Hands-Machine-Learning-Scikit-Learn-Tensorflow/dp/1098125975/ref=pd_lpo_sccl_1/142-4552221-1953647?pd_rd_w=kOFhh&content-id=amzn1.sym.5ca78996-70c7-4a7b-b60c-f030ccc1aa2f&pf_rd_p=5ca78996-70c7-4a7b-b60c-f030ccc1aa2f&pf_rd_r=N3F00572XY4248MV7JMY&pd_rd_wg=6TTPH&pd_rd_r=b410331e-99ad-491e-b629-1f4b85fd8931&pd_rd_i=1098125975&psc=1

IMAGES_PATH = Path() / "images" / "end_to_end_project"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


import matplotlib.pyplot as plt

# extra code – the next 5 lines define the default font sizes
plt.rc("font", size=14)
plt.rc("axes", labelsize=14, titlesize=14)
plt.rc("legend", fontsize=14)
plt.rc("xtick", labelsize=10)
plt.rc("ytick", labelsize=10)

train[["target"]].hist(bins=50, figsize=(12, 8))
save_fig("atarget_histogram_plots")  # extra code
plt.show()

train["target_cat"] = pd.cut(
    train["target"], bins=[0.0, 4.0, 8.0, 12.0, 16.0, np.inf], labels=[1, 2, 3, 4, 5]
)

train["target"].describe()

strat_train_set, strat_test_set = train_test_split(
    train, test_size=0.3, stratify=train["target_cat"], random_state=42
)

percentage1 = strat_test_set["target_cat"].value_counts() / len(strat_test_set)

percentage2 = strat_train_set["target_cat"].value_counts() / len(strat_train_set)

# end of section extracted from Aurélien Géron's repository
##############################
##############################
##############################

strat_test_set.drop("target_cat", axis=1, inplace=True)
strat_train_set.drop("target_cat", axis=1, inplace=True)


shape = strat_test_set.shape
y_test = strat_test_set.pop("target")

y_train = strat_train_set.pop("target")  # train is your submission!

rf = RandomForestRegressor(n_estimators=1000, max_depth=7, n_jobs=-1, random_state=42)
rf.fit(strat_train_set, y_train)

y_hat = rf.predict(strat_test_set)

error = mean_squared_error(y_true=y_test, y_pred=y_hat, squared=False)
print(error)

y_hat = rf.predict(strat_train_set)
error = mean_squared_error(y_true=y_train, y_pred=y_hat, squared=False)

print(error)

cols = train.columns

shape = train.shape

strat_train_set.to_csv(
    "/gh/kaggle-pg-3x21/kaggle-pg-3x21/data/strattified_data.csv", index=False
)
