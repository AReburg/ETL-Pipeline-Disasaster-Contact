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
from nltk.stem.porter import PorterStemmer
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
from sklearn import multioutput
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import f1_score, classification_report, make_scorer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier


def get_all_categories(df):
    categ = [row['categories'].split(',') for index, row in df.iterrows()]
    all_categories = [cat.strip('-1').replace("_", " ").title() for sublist in categ for cat in sublist if cat not in categ]
    return list(set(all_categories))


def load_data(database_filepath):
    """ """
    con = sqlite3.connect(os.path.join(os.path.dirname(__file__), database_filepath))
    df = pd.read_sql_query("SELECT * FROM model_data", con)
    # df = df.iloc[1:400,:]
    # df['message'] = df.apply(lambda x: tokenize(x.message), axis=1)
    X = df.message.values
    y = df[df.columns[2:]]
    categories = y.columns
    return X, y, categories


# https://practicaldatascience.co.uk/machine-learning/how-to-save-and-load-machine-learning-models-using-pickle#:~:text=To%20save%20the%20model%20all,pkl%20.
"""
def tokenize(text):
   #  extract numbers and words and tokenize the messages
  #  stop wards are removed

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
def f1_scorer_eval (y_true, y_pred):
    """A function that measures mean of F1 for all classes
       Returns an average value of F1 for sake of evaluation whether model predicts better or worse in GridSearchCV
    """
    #converting y_pred from np.array to pd.dataframe
    #keep in mind that y_pred should a pd.dataframe rather than np.array
    y_pred = pd.DataFrame (y_pred, columns = y_true.columns)


    #instantiating a dataframe
    report = pd.DataFrame ()

    for col in y_true.columns:
        #returning dictionary from classification report
        class_dict = classification_report (output_dict = True, y_true = y_true.loc [:,col], y_pred = y_pred.loc [:,col])

        #converting from dictionary to dataframe
        eval_df = pd.DataFrame (pd.DataFrame.from_dict (class_dict))

        #dropping unnecessary columns
        eval_df.drop(['micro avg', 'macro avg', 'weighted avg'], axis =1, inplace = True)

        #dropping unnecessary row "support"
        eval_df.drop(index = 'support', inplace = True)

        #calculating mean values
        av_eval_df = pd.DataFrame (eval_df.transpose ().mean ())

        #transposing columns to rows and vice versa
        av_eval_df = av_eval_df.transpose ()

        #appending result to report df
        report = report.append (av_eval_df, ignore_index = True)

    #returining mean value for all classes. since it's used for GridSearch we may use mean
    #as the overall value of F1 should grow.
    return report ['f1-score'].mean ()

def tokenize(text):
    """Tokenization function. Receives as input raw text which afterwards normalized, stop words removed, stemmed and lemmatized.
    Returns tokenized text"""

    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    stop_words = stopwords.words("english")

    #tokenize
    words = word_tokenize (text)

    #stemming
    stemmed = [PorterStemmer().stem(w) for w in words]

    #lemmatizing
    words_lemmed = [WordNetLemmatizer().lemmatize(w) for w in stemmed if w not in stop_words]

    return words_lemmed

def get_metrics(test_value, predicted_value):
    """
    get_metrics calculates f1 score, accuracy and recall

    Args:
        test_value (list): list of actual values
        predicted_value (list): list of predicted values

    Returns:
        dictionray: a dictionary with accuracy, f1 score, precision and recall
    """
    accuracy = accuracy_score(test_value, predicted_value)
    precision = round(precision_score(
        test_value, predicted_value, average='micro'))
    recall = recall_score(test_value, predicted_value, average='micro')
    f1 = f1_score(test_value, predicted_value, average='micro')
    return {'Accuracy': accuracy, 'f1 score': f1, 'Precision': precision, 'Recall': recall}


def multi_class_score(y_true, y_pred):
    accuracy_results = []
    for i, column in enumerate(y_true.columns):
        accuracy = accuracy_score(
            y_true.loc[:, column].values, y_pred[:, i])
        accuracy_results.append(accuracy)
    avg_accuracy = np.mean(accuracy_results)
    return avg_accuracy


def build_model():
    # write custom scoring for multiclass classifier
    # compute bag of word counts and tf-idf values
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize, use_idf=True, smooth_idf=True, sublinear_tf=False)

    # clf = MultiOutputClassifier(RandomForestClassifier(random_state = 42))
    clf = RandomForestClassifier(random_state=42)

    pipeline = Pipeline([('vectorizer', vectorizer), ('clf', clf)])
    score = make_scorer(multi_class_score)
    parameters = {
        'clf__criterion': ['gini', 'entropy'],
        'clf__max_features': ['auto', 'sqrt'],
        'clf__max_depth': [5, 10, 20, 30, 40],
        'clf__random_state': [42]}

    cv_rf_tuned = GridSearchCV(pipeline, param_grid=parameters, scoring=score,
                               n_jobs=-1,
                               cv=5, refit=True, return_train_score=True, verbose=10)

    return cv_rf_tuned
""" latest
def build_model():

    #setting pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', multioutput.MultiOutputClassifier (RandomForestClassifier()))
        ])

    # fbeta_score scoring object using make_scorer()
    scorer = make_scorer (f1_scorer_eval)

    #model parameters for GridSearchCV
    parameters = {  'vect__max_df': (0.75, 1.0),
                    'clf__estimator__n_estimators': [10, 20],
                    'clf__estimator__min_samples_split': [2, 5]
              }
    cv = GridSearchCV (pipeline, param_grid= parameters, scoring = scorer, verbose =7 )

    return cv


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
    model = GridSearchCV(pipeline, param_grid=parameters, cv=3)
    return model


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
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring = total_scorer)
    return cv



def build_model():
    '''
    Output: ML model
    '''
    # build pipeline
    pipeline = Pipeline([

        ('count', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    # set parameters
    parameters = {
        'count__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__min_samples_split': [2, 4],
    }
    # set score metric
    total_scorer = make_scorer(f1_score, average='micro')
    # build model
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring=total_scorer,
                      verbose=3)
    return cv
"""
def display_results(cv, y_test, y_pred):
    labels = np.unique(y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = (y_pred == y_test).mean()

    print("Labels:", labels)
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)
    print("\nBest Parameters:", cv.best_params_)


def evaluate_model(model, X_test, y_test, category_names):
    y_pred = model.predict(X_test)
    try:
        display_results(model, y_test, y_pred)
    except:
        print("error in display results")
        pass
    try:
        class_report = classification_report(y_test, y_pred, target_names=category_names)
        print(class_report)
    except:
        print("error in classification report")


def save_model(model, model_filepath):
    # pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))
    # https://github.com/Kusainov/udacity-disaster-response/blob/master/train_classifier.py
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():

    if len(sys.argv) == 3:
        from time import time
        start =time()
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
        print(f"{time()-start}")

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()