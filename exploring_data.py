import pandas as pd
import zipfile

with zipfile.ZipFile("/gh/kaggle-pg-3x21/data/playground-series-s3e21.zip") as z:
   # open the csv file in the dataset
   with z.open("sample_submission.csv") as f:
    # read the dataset
      train = pd.read_csv(f)
train.to_csv("/gh/kaggle-pg-3x21/sample.csv",index=False)

# kaggle competitions submit -c playground-series-s3e21 -f ./data/sample.csv -m "Message"