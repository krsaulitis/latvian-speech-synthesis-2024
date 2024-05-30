import pandas as pd

model = 'youtube'
df_full = pd.read_csv(f'../data/converted/NISQA_results_{model}_full.csv')
df_tts = pd.read_csv(f'../data/converted/NISQA_results_{model}_tts.csv')

# df_full = pd.read_csv('../data/NISQA_results.csv')

df_full_first_5000 = df_full.iloc[:5000]
df_tts_first_5000 = df_tts.iloc[:5000]

avg_mos = df_tts['mos_pred'].mean()
avg_quality = df_full['mos_pred'].mean()
avg_coloration = df_full['col_pred'].mean()
avg_noisiness = df_full['noi_pred'].mean()
avg_discontinuity = df_full['dis_pred'].mean()
avg_loudness = df_full['loud_pred'].mean()

print(f'MOS: {avg_mos}')
print(f'Quality: {avg_quality}')
print(f'Coloration: {avg_coloration}')
print(f'Noisiness: {avg_noisiness}')
print(f'Discontinuity: {avg_discontinuity}')
print(f'Loudness: {avg_loudness}')
