from flask import Flask, render_template, url_for, request
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import heapq


def summ(text):
    text = str(text)
    text = re.sub(r"\[[0-9]*\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    clean_text = text.lower()
    clean_text = re.sub(r"\W", " ", clean_text)
    clean_text = re.sub(r"\d", " ", clean_text)
    clean_text = re.sub(r"\s+", " ", clean_text)

    # since the clean text is already preprocessed, that cannot be sent tokenized. We want sentences as at the end we wil calculate score for each sentence.
    sentences = nltk.sent_tokenize(text)
    stop_words = set(stopwords.words("english"))

    word2count = {}
    for word in word_tokenize(clean_text):
        if word not in stop_words:
            if word not in word2count.keys():
                word2count[word] = 1
            else:
                word2count[word] += 1

    for key in word2count.keys():
        word2count[key] = word2count[key]/max(word2count.values())

    sent2score = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word2count.keys():
                # to keep the summary more efficient.
                if len(sentence.split(" ")) < 30:
                    if sentence not in sent2score.keys():
                        sent2score[sentence] = word2count[word]
                    else:
                        sent2score[sentence] += word2count[word]

    best_sentences = heapq.nlargest(5, sent2score, key=sent2score.get)

    return " ".join(best_sentences)


app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == "POST":
        s = str(request.form['article'])
        a = summ(s)
        l = [s, a]

        return render_template('summary.html', l=l)

    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
