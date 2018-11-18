# from https://www.kaggle.com/eikedehling/feature-engineering

import pandas as pd

df = pd.read_csv('train.csv')

df['total_length'] = df['comment_text'].apply(len)
df['capitals'] = df['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
df['caps_vs_length'] = df.apply(lambda row: float(row['capitals'])/float(row['total_length']), axis=1)
df['num_exclamation_marks'] = df['comment_text'].apply(lambda comment: comment.count('!'))
df['num_question_marks'] = df['comment_text'].apply(lambda comment: comment.count('?'))
df['num_punctuation'] = df['comment_text'].apply(lambda comment: sum(comment.count(w) for w in '.,;:'))
df['num_symbols'] = df['comment_text'].apply(lambda comment: sum(comment.count(w) for w in '*&$%'))
df['num_words'] = df['comment_text'].apply(lambda comment: len(comment.split()))
df['num_unique_words'] = df['comment_text'].apply(lambda comment: len(set(w for w in comment.split())))
df['words_vs_unique'] = df['num_unique_words']/df['num_words']
df['num_smiles'] = df['comment_text'].apply(lambda comment: sum(comment.count(w) for w in (':-)', ':)', ';-)', ';)')))

features = ('total_length', 'capitals', 'caps_vs_length', 'num_exclamation_marks', 
            'num_question_marks', 'num_punctuation', 'num_symbols', 'num_words',
            'num_unique_words', 'words_vs_unique', 'num_smiles')

columns = ('toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate')

rows = [{c:df[f].corr(df[c]) for c in columns} for f in features]

df_correlations = pd.DataFrame(rows, index=features)

print(df_correlations)