import pandas as pd

data = [
['Japan','Honda','Blue','1980','Economy','Positive'],
['Japan','Toyota','Green','1970','Sports','Negative'],
['Japan','Toyota','Blue','1990','Economy','Positive'],
['USA','Chrysler','Red','1980','Economy','Negative'],
['Japan','Honda','White','1980','Economy','Positive']
]

df = pd.DataFrame(data, columns=['Origin','Manufacturer','Color','Decade','Type','Class'])

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

hypothesis = ['0'] * len(X.columns)

for i in range(len(X)):
    if y[i] == 'Positive':
        for j in range(len(hypothesis)):
            if hypothesis[j] == '0':
                hypothesis[j] = X.iloc[i, j]
            elif hypothesis[j] != X.iloc[i, j]:
                hypothesis[j] = '?'

print("Final Hypothesis:", hypothesis)
