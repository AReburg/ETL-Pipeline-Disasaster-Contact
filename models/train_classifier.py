"""db_name = "data.db"
table_name = "LITTLE_BOBBY_TABLES"

engine = sqlalchemy.create_engine("sqlite:///%s" % db_name, execution_options={"sqlite_raw_colnames": True})
df = pd.read_sql_table(table_name, engine)
at = sqlite3.connect('data.db')

query = dat.execute("SELECT * From <TABLENAME>")
cols = [column[0] for column in query.description]
results= pd.DataFrame.from_records(data = query.fetchall(), columns = cols)

    #with open(path, 'wb') as f: #'mypickle.pickle'
    #    pickle.dump(some_obj, f)
"""
import sys
import pickle
import os
import sqlite3
import pandas as pd
import nltk
#nltk.download(['punkt', 'wordnet'])
import re
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

def unique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
   # cat = [[j for j in row['categories'].split(',') if j not in cat] for index, row in df.iterrows()]
    #for index, row in df.iterrows():
    #    tmp =row['categories'].split(',')
    #    for k in tmp:
    #        if k not in cat:
    #            cat.append(k)

    # output = unique([word for line in fhand for word in line.split()])


def get_all_categories(df):
    categ = [row['categories'].split(',') for index, row in df.iterrows()]
    all_categories = [cat.strip('-1').replace("_", " ").title() for sublist in categ for cat in sublist if cat not in categ]
    return list(set(all_categories))


def load_data(database_filepath):
    """ """
    con = sqlite3.connect(os.path.join(os.path.dirname(__file__), database_filepath))
    df = pd.read_sql_query("SELECT * FROM model_data", con)
    # df['message'] = df.apply(lambda x: tokenize(x.message), axis=1)
    X = df.message.values
    y = df[df.columns[2:]]
    categories = y.columns
    return X, y, categories


# https://practicaldatascience.co.uk/machine-learning/how-to-save-and-load-machine-learning-models-using-pickle#:~:text=To%20save%20the%20model%20all,pkl%20.
def tokenize(text):
    """ extract numbers and words and tokenize the messages
    stop wards are removed """

    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenize
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    # Remove stop words
    clean_tokens = [w for w in clean_tokens if w not in stopwords.words("english")]
    clean_tokens = ",".join(str(x) for x in clean_tokens)
    return clean_tokens

"""
def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    # parameters set to this due to reduce the size of pkl file, which were too large (600MB) for uploading to github with my previous parameters.
    parameters = {
        'clf__estimator__n_estimators': [10],
        'clf__estimator__min_samples_split': [2],

    }
    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4, verbose=2, cv=3)
    return model
"""

def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
        ])),

        ('clf', RandomForestClassifier())
    ])
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'clf__n_estimators': [50, 100, 200],
        'clf__min_samples_split': [2, 3, 4]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv



def display_results(cv, y_test, y_pred):
    labels = np.unique(y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = (y_pred == y_test).mean()

    print("Labels:", labels)
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)
    print("\nBest Parameters:", cv.best_params_)


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    display_results(model, Y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=category_names)
    print(class_report)


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    database_filepath = "../data/DisasterResponse.db"
    print('Loading data...\n    DATABASE: {}'.format(database_filepath))
    X, y, category_names = load_data(database_filepath)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print('Building model...')
    model = build_model()

    print('Training model...')
    model.fit(X_train, y_train)

    print('Evaluating model...')
    evaluate_model(model, X_test, y_test, category_names)

    print('Saving model...\n    MODEL: {}'.format('test3.pkl'))
    save_model(model, 'test3.pkl')


    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)


        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()