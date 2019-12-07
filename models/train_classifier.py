
# import libraries
import sys
import re
import nltk
import warnings
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import joblib

warnings.simplefilter(action='ignore', category=FutureWarning)
nltk.download('wordnet')

def load_data(DB_path):
    """
    this function take the path of database sqlite and load it to data frame
    then split the data into Messages and categories 
    """
    print("loading the database")
    # load data from database
    engine = create_engine('sqlite:///{}'.format(DB_path))
    #print(engine)
    df = pd.read_sql_table("Msgs", con = engine)
    print("the data  successfully loaded :)")
    # or using read_sql function
    #df=pd.read_sql(sql="SELECT * FROM Msgs",con=engine)
    X= df['message']
    Y=df.drop(["id","message","genre","original"],axis=1)

    return X,Y
    
    

def tokenize(text):
    """
    this function take the text and perform text processing (Normalization and Tokenization)
    then returns the text which cleaned
    """
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


def train_model(pipeline,X_train, y_train):
    """
    this function take the training data and train the data pipeline model
    and return the trained model
    """
    model=pipeline
    print("Start training ....") 
    # train classifier
    model=model.fit(X_train, y_train)
    print("Training is finished ....")
    print("Now the model ready to evaluation") 

    return model

    

def evaluate_model(model,X_test,y_test):
    """
    this function take the trained model and evaluate it on the test data
    and print the Accuracy and classification report
    
    """
    # print classification_report
    print("Evaluation the model .....")
    y_pred = model.predict(X_test)
    print('Accuracy: {}'.format(np.mean(y_test.values == y_pred)))
    print("Classification report:")
    print(classification_report(y_test.values, y_pred, target_names=y_test.columns))
    



def run_pipeline():
    """
    Build the data pipeline
    and return the gridsearch model
    """
    #Extract features using Bag of words,transforms the data and select the Random forest algorithm as classifire
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(DecisionTreeClassifier()))])
    #use dic to store the parameters of algorithm
    parameters = {
    "clf__estimator__splitter":["best","random"]
    ,"clf__estimator__criterion":["gini","entropy"]
    
    }   
    #build the grid seach 
    cv = GridSearchCV(pipeline,param_grid=parameters,cv=3)
    #run thr pipeline
    print("Now the data pipeline ready to train on the data :)")

    return cv

def save_model(model,path):
    """
    this function take the path and name of model 
    and save the model in the path 
    """
 
    joblib.dump(model, path)
    
    print("The model successfully saved :) ")
    
    


def main():

    if len(sys.argv)==3:
        #get the paths of database and name of model from the terminal 
        DB_path = sys.argv[1]  # get path of Database
        model_path=sys.argv[2] # get path of model
     
        #1.Load the data
        X,Y=load_data(DB_path)
        
        #2.Split data to train and test
        X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.30, random_state=42)

        #3.Build the Data pipeline
        grid_model=run_pipeline()  # run data pipeline

        #4.Train the model
        trained_model=train_model(grid_model,X_train, y_train)
        #5.Evaluate the model
        evaluate_model(trained_model,X_test,y_test)
        #6.Save the model
        save_model(trained_model,model_path)
    else:
        print("Check the number of arguments")
        print("you should enter two arguments like this>>")
        print("path of database like this >> ../data/DisasterResponse.db")
        print("name of classifire like this >> classifier.pkl")
        print("In the end you have like this in the terminal>>")
        print("python train_classifier.py ../data/DisasterResponse.db classifier.pkl")



if __name__ == '__main__':
   main()

