import streamlit as st
import pickle as pk
from nltk.corpus import stopwords
import string
import nltk
from nltk.stem.porter import PorterStemmer

tfidf=pk.load(open('./Pickle_files/vectorizer.pkl', 'rb'))
model=pk.load(open('./Pickle_files/model.pkl', 'rb'))
nltk.download('punkt_tab')
nltk.download('stopwords')
def transform_text(text):
    #Convert to lower case
    lower_text=text.lower()
    #Tokenization
    tokenized_text=nltk.word_tokenize(lower_text)
    #Remove special characters
    alnum_words=list()
    for word in tokenized_text:
        if word.isalnum():
            alnum_words.append(word)
    words=list()
    #Removing Stop Words and Punctuation
    for word in alnum_words:
        if word not in stopwords.words('english') and word not in string.punctuation:
            words.append(word)
    #stemming
    ps=PorterStemmer()
    temp=list()
    for word in words:
        temp.append(ps.stem(word))
    words=temp
    return " ".join(words)

st.title("Email/SMS Spam Predictor")

input_msg=st.text_area("Enter the message")

if st.button(label="Check"):
    if(input_msg):
        transformed_msg=transform_text(input_msg)
        vector_msg=tfidf.transform([transformed_msg])
        prediction=model.predict(vector_msg)    
        if(prediction==0):
            st.success("The message is HAM...")
        else:
            st.warning("The message is SPAM!")



