
# import packages
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(data_file,categories_file,DB_file):
    """
    this function take the paths of dataset,categories and database
    and clean the data then save the new data into sqlite table
    """
    # read in file
    #print(data_file)
    #print(type(data_file))
    messages = pd.read_csv(data_file)
    categories = pd.read_csv(categories_file)
    
    # merge datasets
    df = messages.merge(categories,on='id')
    #Clean the data
    df=Clean_data(df)
    # load to database
    engine = create_engine("sqlite:///{}".format(DB_file))
    #save the cleaned data as DB table 
    df.to_sql(name ="Msgs", con=engine, index=False,if_exists='replace')


def Clean_data(Data):
    """
    this function take the data
    and returns cleaned data
    """
    # create a dataframe of the 36 individual category columns
    categories = Data["categories"].str.split(pat=";",expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    clean_categories=lambda row:[item[:-2] for item in row]
    category_colnames = clean_categories(row)
    #print(category_colnames)
    # rename the columns of `categories`
    categories.columns = category_colnames

    #Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    #Replace two values to ones
    categories.related.replace(2, 1, inplace=True) 

    #Replace categories column in df with new category columns.
    # drop the original categories column from `df`
    Data = Data.drop(columns="categories")  
    # concatenate the original dataframe with the new `categories` dataframe
    Data[categories.columns] = categories

    # Remove duplicates
    Data.drop_duplicates(inplace=True)

    return Data

def main():
    #print(len(sys.argv))
    if len(sys.argv)==4:
        data_file = sys.argv[1]  # get filename of dataset
        categories_file=sys.argv[2] 
        DB_file=sys.argv[3]
        load_data(data_file,categories_file,DB_file)  # run data pipeline

    else:
        print("Please check the arguments must as script_name.py data.csv categories.csv DB_file.db")

       



if __name__ == '__main__':
    main()




