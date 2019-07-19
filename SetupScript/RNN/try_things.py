import pandas as pd

df = pd.DataFrame(columns = ['user_id', 'score', 'score2'])

print(df)

values = ['ciccio', '30']
valuess = []
valuess.append(values)
print(valuess)
row = pd.DataFrame([['ciccio', 30, 31]], columns = ['user_id', 'score', 'score2'])
df = df.append(row)

row = pd.DataFrame([['ciccio', 15, 31]], columns = ['user_id', 'score', 'score2'])
df = df.append(row)

row = pd.DataFrame([['ciccio', 22, 31]], columns = ['user_id', 'score', 'score2'])
df = df.append(row)

row = pd.DataFrame([['piccio', 30, 31]], columns = ['user_id', 'score', 'score2'])
df = df.append(row)

row = pd.DataFrame([['piccio', 56, 31]], columns = ['user_id', 'score', 'score2'])
df = df.append(row)

df = df.groupby('user_id').apply(lambda x: x.sort_values(['score'], ascending = False))


print(df)