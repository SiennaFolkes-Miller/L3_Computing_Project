import pandas as pd

df = pd.read_csv('supernovae_data.csv', delim_whitespace=True, comment='#')           
print(df)

split_line = 45
high_redshift = df.iloc[:split_line]
low_redshift = df.iloc[split_line:]

low_redshift_values = low_redshift[['redshift', 'm_effective', 'error']].values  

high_redshift_values = high_redshift[['redshift', 'm_effective', 'error']].values  

print(low_redshift_values[1])