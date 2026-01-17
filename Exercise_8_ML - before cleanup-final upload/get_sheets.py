import pandas as pd
xls = pd.ExcelFile('curse-of-dimensionality.xlsx')
print(xls.sheet_names)
