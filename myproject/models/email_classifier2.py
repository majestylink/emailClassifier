import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os

df = pd.read_csv(r'C:\Users\SAMUEL\Downloads\spam or ham.csv', encoding='latin-1')
df['spam'] = df['mail type'].apply(lambda x: 1 if x == 'spam' else 0)
# print(df.head())

X_train, X_test, y_train, y_test = train_test_split(df.message, df.spam, test_size=0.2)

model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

model.fit(X_train, y_train)
print(model.score(X_test, y_test))


pickle.dump(model, open('myModel.pickel', 'wb'))
print(os.getcwd())

print(model.predict([
    "Hey Flamezee, can we go together to watch football game tomorrow",
    "Upto 20% discount on parking, exclusive offer just you. don't miss this reward!",
    "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question("
    "std txt rate)T&C's apply 08452810075over18's "
]))
