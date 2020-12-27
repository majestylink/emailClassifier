import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os

df = pd.read_csv(r'C:\Users\SAMUEL\Downloads\spam or ham.csv', encoding='latin-1')
# print(df.head())

# print(df.groupby('mail type').describe())

df['spam'] = df['mail type'].apply(lambda x: 1 if x == 'spam' else 0)
# print(df.head())

X_train, X_test, y_train, y_test = train_test_split(df.message, df.spam, test_size=0.25)
v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)
# print(X_train_count.toarray()[:3])

model = MultinomialNB()
model.fit(X_train_count, y_train)

pickle.dump(model, open('themodel.pickle', 'wb'))
pickle.dump(v, open('thescaler.pickle', 'wb'))

emails = [
    "Hello sweetheart, I don't know if you'll be free tonight so that we can go to the cinema together",
    "Get up to 50% discount by clicking on the link below"
]
emails_count = v.transform(emails)
print(model.predict(emails_count))

a = model.predict(emails_count)
for x in a:
    if x == 1:
        print('This email is a spam')
    else:
        print('This email is a ham')
#
X_test = v.transform(X_test)
print("Our model score is", model.score(X_test, y_test))
#
# print(type(emails), type(emails_count))

