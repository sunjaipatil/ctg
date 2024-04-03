import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.signal import welch
import modules, glob
files_bad = glob.glob('ctg/FHR/BAD_BABYURN/*')
files_good = glob.glob('ctg/FHR/GOOD_BABYURN/*')
files_mod = glob.glob('ctg/FHR/Moderate_BABYURN/*')
files = files_bad + files_good +files_mod

df = pd.DataFrame(index = sorted(files), columns = ['% Area'])

files = sorted(files)

for file_path in files:
    result = modules.uc_sensitivity(file_path)
    df.at[file_path, '% Area'] = result
    print(result)
    


__import__("IPython").embed()


