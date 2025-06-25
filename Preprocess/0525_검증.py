import pandas as pd
import os
from pathlib import Path

PATH=Path(os.path.abspath(__file__)).parent

df=pd.read_csv(PATH / "solution_X.csv")
print(df)