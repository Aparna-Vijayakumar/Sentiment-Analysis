from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from collections import Counter
from nltk.stem.porter import PorterStemmer

columns = ['review','rating']
df = pd.DataFrame(columns=columns)

with open('./data/unprocessed/sorted_data/beauty/all.review', 'r', encoding = 'latin-1') as infile:
    copy = False
    i = 0
    for line in infile:
        if line.strip() == "<review>":
            begin = True
            rating = []
            rev = []
            i = i+1
        elif line.strip() == "</review>":
            begin = False
            df.loc[i] = [rev, rating]
        elif begin and line.strip() == "<review_text>":
            copy = True
            marker = True
            continue
        elif begin and line.strip() == "</review_text>":
            copy = False
            continue
        elif begin and line.strip() == "<rating>":
            copy = True
            marker = False
            continue
        elif begin and line.strip() == "</rating>":
            copy = False
            continue
        elif copy and marker:
            rev.append(line)
        elif copy == True and marker == False:
            rating.append(line)

stop_words = set(stopwords.words('english'))
porter = PorterStemmer()
for index, row in df.iterrows():
    data_rating = row['rating']
    if float(data_rating[0]) < 3:
        df.at[index,'rating'] = -1
    elif float(data_rating[0]) > 3:
        df.at[index, 'rating'] = 1
    else:
        df.at[index, 'rating'] = 0
        print(df.review)
    data = row['review']
    tokens = word_tokenize(data[0])
    words = [word for word in tokens if word.isalnum()]
    words = [w for w in words if not w in stop_words]
    df.at[index,'review'] = [porter.stem(word) for word in words]

df = df[df.rating != 0]
df_counts = pd.DataFrame([Counter(x) for x in df['review']]).fillna(0).astype(int)
processed_data = df_counts.to_csv (r'./data/processed/beauty.csv', index = None, header=True)
labels = df['rating'].to_csv(r'./data/processed/beautylabels.csv', index = None, header=True)
