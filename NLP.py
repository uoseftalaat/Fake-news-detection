import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from sklearn import model_selection, preprocessing, linear_model,  metrics
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import WordNetLemmatizer
from joblib import dump

def preprocess(data,target_column):
    data[target_column] = data[target_column].apply(lambda x: " ".join(x.lower() for x in x.split()))
    stop = stopwords.words('english')
    data[target_column] = data[target_column].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    st = PorterStemmer()
    data[target_column] = data[target_column].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
    lemmatizer = WordNetLemmatizer()
    data[target_column] = data[target_column].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))
    
    
    
def train_model(classifier, feature_vector_train, label_train, feature_vector_valid,label_valid):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label_train)
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    return metrics.accuracy_score(predictions, label_valid)


news = pd.read_csv("news.csv")
news['class'] = np.where(news['label'] == 'FAKE', 0, 1)
news = news[['text', 'label']]
news.drop_duplicates(inplace=True)

preprocess(news, 'text')

text = ' '.join(news['text'].tolist())
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)


# Create a bar chart of the most common words in the dataset
word_counts = Counter(text.split())
top_words = word_counts.most_common(10)

plt.figure(figsize=(16, 12))
sns.barplot(x=[w[0] for w in top_words], y=[w[1] for w in top_words])
plt.title('Top 20 Most Common Words')
plt.xlabel('Words')
plt.ylabel('Count')
plt.show()

# Create a scatter plot of the word counts for each article
article_word_counts = [len(article.split()) for article in news['text'].tolist()]

plt.figure(figsize=(16, 12))
sns.scatterplot(x=range(len(article_word_counts)), y=article_word_counts)
plt.title('Word Counts for Each Article')
plt.xlabel('Article Number')
plt.ylabel('Word Count')
plt.show()




train_x, valid_x, train_y, valid_y = model_selection.train_test_split(news['text'], news['label'])

encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(news['text'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)
LR = linear_model.LogisticRegression()
accuracy = train_model(LR, xtrain_tfidf, train_y, xvalid_tfidf,valid_y)
print ("Accuracy: ", accuracy)
MN = MultinomialNB()
accuracy1 = train_model(MN , xtrain_tfidf, train_y, xvalid_tfidf,valid_y)
print("Accuracy: ", accuracy1)
dump(LR, 'LR.joblib')
dump(MN, 'MN.joblib')
#the accuracy plot
plt.figure(figsize=(16, 12))
sns.barplot(x=['LR' , 'Naive bayes'], y=[accuracy,accuracy1])
plt.title('accuracy')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.show()













