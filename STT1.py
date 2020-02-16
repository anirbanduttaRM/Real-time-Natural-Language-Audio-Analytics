# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 20:12:01 2020

@author: 138410
"""
__author__ = 'anirbandutta'

#import pprint
#import sys
#for PyAudio
#pip install pipwin
#pipwin install pyaudio
#https://stackoverflow.com/questions/53866104/pyaudio-failed-to-install-windows-10
#pprint.pprint(sys.path)
import numpy as np
import speech_recognition as sr

r = sr.Recognizer()
m = sr.Microphone()

import pandas as pd
import nltk
import os
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.downloader.download('vader_lexicon')
final=0
txt=''

def analyze_sentiment_vader_lexicon(review, 
                                    threshold=0.1,
                                    verbose=False):

    global final
    # analyze the sentiment for review
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(review)
    # get aggregate scores and final sentiment
    agg_score = scores['compound']
    final_sentiment = 'positive' if agg_score >= threshold\
                                   else 'negative'
    if verbose:
        #display detailed sentiment statistics
        positive = str(round(scores['pos'], 2)*100)+'%'
        final = round(agg_score, 2)
        negative = str(round(scores['neg'], 2)*100)+'%'
        neutral = str(round(scores['neu'], 2)*100)+'%'
        sentiment_frame = pd.DataFrame([[final_sentiment, final, positive,
                                        negative, neutral]],
                                        columns=pd.MultiIndex(levels=[['SENTIMENT STATS:'], 
                                                                     ['Predicted Sentiment', 'Polarity Score',
                                                                       'Positive', 'Negative', 'Neutral']], 
                                                              labels=[[0,0,0,0,0],[0,1,2,3,4]]))
    return final

from nltk.tag import pos_tag
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
#sentence = 'OVERALL, GREAT SERVICE.'
def analyze_polarising_words(sentence):
    
    tokenized_sentence = nltk.word_tokenize(sentence)
    print(sentence)
    print(tokenized_sentence)

    sid = SentimentIntensityAnalyzer()
    pos_word_list=["Positive:"]
    neu_word_list=["Neutral:"]
    neg_word_list=["Negative:"]

    for word in tokenized_sentence:
        if (sid.polarity_scores(word)['compound']) >= 0.1:
            pos_word_list.append(word)
        elif (sid.polarity_scores(word)['compound']) <= -0.1:
            neg_word_list.append(word)
        else:
            neu_word_list.append(word)    
            
    result = pos_word_list + neg_word_list + neu_word_list  
    print(result)
    return result

try:
    print("A moment of silence, please...")
    with m as source:
        r.adjust_for_ambient_noise(source)
        print("Set minimum energy threshold to {}".format(r.energy_threshold))
        while True:
            print("Say something!")
            audio = r.record(source, 10) #Here's where I made the change
            print("Got it! Now to recognize it...")
            try:
                # recognize speech using Google Speech Recognition
                value = r.recognize_google(audio) 
                if str is bytes: # this version of Python uses bytes for strings (Python 2)
                    print(u"You said {}".format(value).encode("utf-8"))
                    ss = analyze_sentiment_vader_lexicon(format(value).decode("utf-8"), threshold=0.1,verbose=True)
                    polarizing_words = analyze_polarising_words(format(value))
                    #print(ss)
                    txt=format(value).encode("utf-8")
                else: # this version of Python uses unicode for strings (Python 3+)
                    print("You said {}".format(value))
                    ss = analyze_sentiment_vader_lexicon(format(value), threshold=0.1,verbose=True)
                    polarizing_words = analyze_polarising_words(format(value))                    
                    txt=format(value)
            except sr.UnknownValueError:
                print("Oops! Didn't catch that")
            except sr.RequestError as e:
                print("Uh oh! Couldn't request results from Google Speech Recognition service; {0}".format(e))
            tagged_sent = pos_tag(txt.split())
            # [('Michael', 'NNP'), ('Jackson', 'NNP'), ('likes', 'VBZ'), ('to', 'TO'), ('eat', 'VB'), ('at', 'IN'), ('McDonalds', 'NNP')]
            propernouns = [word for word,pos in tagged_sent if pos == 'NNP']
            nouns = [word for word,pos in tagged_sent if pos == 'NN']
            verbs = [word for word,pos in tagged_sent if pos == 'VB']
            print('Polarity score:'+ str(ss))
            print('Polarizing words:'+','.join(polarizing_words))
            print('Keywords:'+','.join(propernouns))
            print('Context:'+','.join(nouns))
            print('Action Words:'+','.join(verbs))

except KeyboardInterrupt:
    sys.exit()