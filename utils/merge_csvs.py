import os
import pandas as pd
import sys


FILE_START = sys.argv[1]
ROOT_PATH, FILE_START = os.path.split(FILE_START)
filename_list = os.listdir(str(ROOT_PATH))
filename_list = [filename for filename in filename_list if filename.startswith(FILE_START)]

OUTPUT_PATH = os.path.join(ROOT_PATH, FILE_START+".csv")

result = pd.DataFrame(columns=['log'])
for filename in filename_list:
    df = pd.read_csv(os.path.join(ROOT_PATH, filename))
    result = result.merge(df, on='log', how='outer') 
    print(df.shape)
result.to_csv(OUTPUT_PATH, index=False)
print(f"Saved dataframe with {result.shape} in {OUTPUT_PATH}")
