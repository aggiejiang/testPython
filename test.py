import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
​from nltk.corpus import stopwords
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def preprocessing(text):
    
    text = text.lower()    #lowercase
    text = re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "URL", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"_", " ", text)
    text = re.sub(r"@\w+", "USER", text)
    text = re.sub(r"#\w+", "TOPIC", text)
    text = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "NUMBER", text)
    text = re.sub(r"[^a-zA-Z0-9\s]","",text)   #remove all non-alphanumeric characters except spaces and single star '
    
    #text = re.sub(r"#\S+", lambda hashtag: " ".join(word_tokenize(hashtag.group()[1:])), text) # segment hastags
    
    # remove stop words
    token = word_tokenize(text)
    filtered_text = []
    filtered_text = [w for w in token if not w in stopwords.words('english')]
    text = " ".join(filtered_text)
    
    #text = re.sub('[a-z]+',lemmatize_word,text)   #lemmatizing words
​
    return text


if __name__ == "__main__":
    
    data = pd.read_csv('english_dataset_sample.csv', encoding = 'gbk')
    
    text = data['text']
    label1 = np.array(data['task_1'])
    
    #print('-------preprocessing-------')
    pre_text = [preprocessing(text[i]) for i in range(len(text))]

    length = int(len(pre_text) * 0.8)
    x_train = list(pre_text[:length])
    y_train = list(label1[:length])
    x_test = list(pre_text[length:])
    y_test = list(label1[length:])
    
    
    #print('--------ngram feature extraction-------')

    #convert words in text to word frequency matrix, a[i][j] indicate the frequency of word j in text i
    count_vectorizer = CountVectorizer()

    #calculate tf-idf value of each word
    tfidf_transformer = TfidfTransformer()

    # 1st fit_transform is to calculate tf-idf
    # 2nd fit_transform is to convert texts into word frequency matrix
    x_train = tfidf_transformer.fit_transform(count_vectorizer.fit_transform(x_train))
    x_test = tfidf_transformer.transform(count_vectorizer.transform(x_test))
    y_train = count_vectorizer.fit_transform(y_train)
    
    #print('-------build model-------')

    classifier = OneVsRestClassifier(LogisticRegression())
    classifier.fit(x_train,y_train)

    y_predicted = classifier.predict(x_test)
    #print(y_predicted.toarray())

    scores = cross_val_score(classifier, x_train, y_train, cv=10)
    print(scores)
    