from textblob import TextBlob
from sklearn import linear_model, preprocessing,model_selection
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow 
import pandas as pd
import numpy as np
from textblob import Word
print("Enter the sentence you want to learn whether it is positive or negative.\n")#Data can be extracted from any source, database or a csv file, but I preferred this :D
text_orginal= input()
text_df= pd.DataFrame([text_orginal],columns=["Text"])
text_series= text_df["Text"].apply(lambda x: " ".join(x.lower() for x in x.split())) #conversion of words to lowercase
text_series = text_series.str.replace("[^\w\s]","",regex= True) #deleting punctuation marks
text_series = text_series.str.replace("\d","",regex=True) #deleting numbers
text_df= pd.DataFrame(text_series,columns=["Text"])
sw = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
      'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 
      'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 
      'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 
      'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 
      'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 
      'just', 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain']
text_series = text_df["Text"].apply(lambda x: " ".join(x for x in x.split() if x not in sw)) #deleting stopwords
text_df= pd.DataFrame(text_series,columns=["Text"])
text = text_series.to_string(index= False)
words = text.split()
if len(words) < 20:
    delete = pd.Series(" ".join(text_df["Text"]).split()).value_counts()[-3:]
    text_series = text_df["Text"].apply(lambda x: " ".join(x for x in x.split() if x not in delete)) #deleting rare wordsdeleting rare words
else:
    delete = pd.Series(" ".join(text_df["Text"]).split()).value_counts()[-10:]
    text_series = text_df["Text"].apply(lambda x: " ".join(x for x in x.split() if x not in delete)) #deleting rare wordsdeleting rare words
text_df= pd.DataFrame(text_series,columns=["Text"])
text_series= text_df["Text"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))#lemmatization
text_df= pd.DataFrame(text_series,columns=["Text"])
df = pd.read_csv("org_opmin.csv",usecols=["Text","Label"])
train_x, test_x, train_y, test_y = model_selection.train_test_split(df["Text"],df["Label"], random_state= 42)
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
test_y = encoder.fit_transform(test_y)
vectorizer = CountVectorizer()
vectorizer.fit(train_x)
x_train_count = vectorizer.transform(train_x) #This process creates a vector for each word and applies it to each line.
loj_model = linear_model.LogisticRegression(solver="liblinear",C=1,intercept_scaling=0.1,max_iter=100,penalty="l2",tol=1)
loj_model = loj_model.fit(x_train_count,train_y)
feedback = loj_model.predict(vectorizer.transform(text_series))
feedback = np.array2string(feedback)
if feedback == "[0]":
    print("\nTHİS SENTENCE IS NEGATIVE")
    feedback = "Negative"
else:
    print("\nTHİS SENTENCE IS POSITIVE")
    feedback = "Positive"
print("Did I guess correctly? Y/N")
conf2 = input()
print("Do you want to save the data? Y/N")
conf1 = input()
if conf1 == "Y":
    if conf2 == "Y":
        new_df = pd.read_csv("opmin2.csv",usecols=["Text","Label"])
        new_data = {"Text": text, "Label": feedback}
        new_df.loc[len(new_df)] = new_data
        new_df.to_csv("opmin2.csv")
        new_org_df = pd.read_csv("org_opmin2.csv",usecols=["Text","Label"])
        label_df = pd.DataFrame([feedback],columns=["Label"])
        df_df= pd.concat([text_df,label_df],ignore_index= True,axis=1,)
        df_df.columns =["Text","Label"]
        new_org_df = pd.concat([new_org_df,df_df],ignore_index= True,axis=0)
        new_org_df.to_csv("org_opmin2.csv")