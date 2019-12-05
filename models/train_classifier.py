
# import libraries
import sys
from sqlalchemy import create_engine
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,recall_score,precision_score
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
nltk.download('wordnet')

def load_data(DB_path):
    # load data from database
    engine = create_engine('sqlite:///{}'.format(DB_path))
    #print(engine)
    df = pd.read_sql_table("Msgs", con = engine)
    # or using read_sql function
    #df=pd.read_sql(sql="SELECT * FROM Msgs",con=engine)
    X= df['message']
    Y=df.drop(["id","message","genre","original"],axis=1)
    #Replace two values to ones
    Y.related.replace(2, 1, inplace=True) 

    return X,Y 

def tokenize(text):
    #1.Punctuation Removal step and normalize
    text=re.sub(r"[^a-zA-Z0-9]"," ",text.lower())
    #print(text)
    #2.Tokenize statments to words
    text=word_tokenize(text)
    #print(text)
    #3.Stop words removal
    text=[w for w in text if w not in stopwords.words("english")]
    #print(text)
    #4.Reduce words to their stems
    stemm_text= [PorterStemmer().stem(w) for w in text]
    #print(stemm_text)
    #5.Reduce words to their root form
    lemmed_text = [WordNetLemmatizer().lemmatize(w) for w in stemm_text]
    #print(lemmed_text)
    return lemmed_text


def train_model(run_pipeline,X_train, y_train):
    #run thr pipeline
    model=run_pipeline()
    print("Start training ....") 
    # train classifier
    model=model.fit(X_train, y_train)
    print("Training is finished ....")
    print("Now the model ready to evaluation") 

    return model

    

def evaluate_model(model,X_test,y_test):
    # print accuracy score & precision and recall 
    y_pred = model.predict(X_test)
    print('Accuracy: {}'.format(accuracy_score(y_test,y_pred)))
    print('nRecall score: {}'.format(accuracy_score(y_test,y_pred)))
    print('Precision score: {}'.format(accuracy_score(y_test,y_pred)))
    



def run_pipeline():
    #Extract features using Bag of words,transforms the data and select the Random forest algorithm as classifire
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    #use dic to store the parameters of algorithm
    parameters = {
    "clf__estimator__n_estimators":[150,200]
    ,"clf__estimator__criterion":["gini","entropy"]
    
    }   
    #build the grid seach 
    cv = GridSearchCV(pipeline,param_grid=parameters)
    print("Now the data pipeline ready to train on the data :)")
    return cv

def save_model(model,path):
    with open(path, 'wb') as file:
        pickle.dump(model, file)
    
    print("The model successfully saved :) ")
    
    



if __name__ == '__main__':
    if len(sys.argv==3): 
        DB_path = sys.argv[1]  # get path of Database
        model_path=sys.argv[2] # get path of model
        #1. load the data
        X,Y=load_data(DB_path)
        grid_model=run_pipeline()  # run data pipeline
        #split data to train and test
        X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.30, random_state=42) 
        trained_model=train_model(run_pipeline,X_train, y_train) 
        evaluate_model(trained_model,X_test,y_test)
        save_model(trained_model,model_path)
    else:
        print("Check the number of arguments")
        print("you should enter two arguments like this>>")
        print("path of database like this >> ../data/DisasterResponse.db")
        print("name of classifire like this >> classifier.pkl")
        print("In the end you have like this in the terminal>>")
        print("python train_classifier.py ../data/DisasterResponse.db classifier.pkl")


