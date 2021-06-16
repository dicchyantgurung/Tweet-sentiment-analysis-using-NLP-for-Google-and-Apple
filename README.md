# Tweet sentiment using NLP for Apple and Google

![Title](https://github.com/dicchyant84/NLP-of-tweet-sentiment-for-Google-and-Apple/blob/main/Images/Title.jpg)

### Business case

Apple and Google are looking to analyze what people are saying about their brand in twitter. People use twitter to share their impulsive yet honest thoughts and opinions. This can provide brands a good idea on what consumers are really feeling about their products. 

In order to achieve this, we will take the following steps:

### EDA:
* Explore the contents of the tweet to identify the product and sentiment of the tweet.
* Analyze the sentiments to identify common issues or contentment expressed by the user.
* Make some meaningful recommendations to the brands that they can use to serve their customers better.

### Modelling:
* Build a model that can classify the tweets into positive, negative or neutral emotions.
* Deploy the model for production to track customer sentiments in real-time.

------------------------------------------------------------------------------------------------

Natural Language Processsing is a branch of artificial intelligence where a computer learns to interacts with the human language. There are various techniques on how we can go about decoding the human language into bits and bytes. Here we will use a simple tokenization process to break down each word and convert them into arrays of intergers that the computer understands.

Using nltk and some code, we are able to easily tokenize the tweets into its simplest form.

```
# Download stopwords from nltk
nltk.download('stopwords')

# Create a list of stopwords and add punctuation to remove them both
stopwords_list = stopwords.words('english')
stopwords_list += list(string.punctuation)

tokenized_tweet = []

# Loop in to access individual tweets
for i in df_clean.index:
    tweet = df_clean.tweet_text[i]
        
    # tokenize each tweet
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    tweet_tokens_raw = nltk.regexp_tokenize(tweet, pattern)
        
    # lowercase each token
    tweet_tokens = [word.lower() for word in tweet_tokens_raw]
        
    # remove stopwords and punctuations
    tweet_words_stopped = [word for word in tweet_tokens if word not in stopwords_list] 
    
    # append the tokens into tokenized_tweets
    tokenized_tweet.append(tweet_words_stopped)
```
Once we have the tokenized text ready, we can move into the analysis part of this EDA.

----------------------------------------------------------------------------------------------------------------------------------

### EDA 1 -  Explore the contents of the tweet to identify the product and sentiment of the user

By creating lists of products that points to different brands, we are able to quickly identify what brand is the tweet directed towards. This can then be plotted on a histogram to check the frequency of tweets for each brand or product.

![Brand mentions](https://github.com/dicchyant84/NLP-of-tweet-sentiment-for-Google-and-Apple/blob/main/Images/Brand_Mentions.jpg)

Overall, Apple seems to have more of a brand presence than Google.
The frequency distribution of tweets for each product can be seen below.

![Product Mentions](https://github.com/dicchyant84/NLP-of-tweet-sentiment-for-Google-and-Apple/blob/main/Images/Product_mentions.jpg)

Taking a closer look at apple products:
* Ipad seems to be the most talked about in out dataset.
* The word Apple comed second followed closely by Iphone.

![Apple Mentions](https://github.com/dicchyant84/NLP-of-tweet-sentiment-for-Google-and-Apple/blob/main/Images/Apple_Mentions.jpg)

Additionally, we can also plot the overall distribution of sentiments across all products as shown below.

![Overall Sentiment](https://github.com/dicchyant84/NLP-of-tweet-sentiment-for-Google-and-Apple/blob/main/Images/Sentiment_distribution.jpg)

The plot above suggests that majority of the tweets in our dataset are of neutral sentiment. All products alse have more positive tweets than negative ones, which is a good indication that the customers are mostly in favour of both brands.

-------------------------------------------------------------------------------------------------------------

### EDA 2 - Analyze the sentiments to identify common issues or contentment expressed by the user

The computer does not understand the meaning of different words even though it is able to differentiate it through a vectorization process. POS tagging which stands for Part-of-Speech tagging is a method used in NLP to tag each word into it's gramatical category or part of the sentence it's in. Here, we will generate POS tags using nltk to distinguish the words and grab the one's that express emotions for the product.

Using some code, we are able to pull the most common words used in each tweet for each product and emotion.
The results are seen below.

Apple positive:

![Apple positive](https://github.com/dicchyant84/NLP-of-tweet-sentiment-for-Google-and-Apple/blob/main/Images/apple_postive_wordcloud.jpg)

Apple negative:

![Apple negative](https://github.com/dicchyant84/NLP-of-tweet-sentiment-for-Google-and-Apple/blob/main/Images/apple_negative_wordcloud.jpg)

Google positive:

![Google positive](https://github.com/dicchyant84/NLP-of-tweet-sentiment-for-Google-and-Apple/blob/main/Images/google_positive_wordcloud.jpg)

Google negative:

![Google negative](https://github.com/dicchyant84/NLP-of-tweet-sentiment-for-Google-and-Apple/blob/main/Images/google_negative_wordcloud.jpg)

In summary:
* Customers seem to be praising both brands for the most part. 
* There are some concerns raised for Apple, but nothing iminent for Google.

### Recommendations for Apple:

* Apple is pretty well-known for all it’s products and is referred to as the ‘cooler’ one of the two.
* Customers show little to no interest in macs and macbooks. However, there seems to be some potential in iTunes.
* Product development or differentiation in the media service industry could help Apple capture more people towards it’s brand.

### Recommendations for Google:

* Google has an overall strong presence as a major tech company.
* However, it needs to engage more with the customers to maintain brand loyalty.

----------------------------------------------------------------------------------------------------------


# Tweet Sentiment Predictor


![LSTM Architecture](https://github.com/dicchyant84/NLP-of-tweet-sentiment-for-Google-and-Apple/blob/main/Images/Architecture-EncoderDecoder_v2-1080x453.png)


Recurring Neural Networks using the LSTM architecture usually perform very well with text classification. We will deploy this model with L1 regularization and 'sigmoid' activation function to train and predict the class.


```
model6 = Sequential()
# Add the embedding layer
model6.add(layers.Embedding(max_words, 40))

# Add the LSTM layer with dropout
model6.add(layers.LSTM(20, dropout=0.3)) 

# Add second layer using relu activation with l2 regularization
model6.add(layers.Dense(10, kernel_regularizer=regularizers.l1(0.005), activation='relu'))

# Final layer using sigmoid
model6.add(layers.Dense(3, activation='sigmoid'))

# Compile the model
model6.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])

# Inspect the model
model6.summary()
```


After couple of iterations to the model, our final model has performed averagely at 65.44% test accuracy. The train accuracy is at 85%. This difference between training and test accuracy suggests that our model is overfitting. Trying different loss fuctions and adding multiple layers also does not seem to improve our model. We can conclude that our data is just not enough to train our model effectively enough to predict with better results.


------------------------------------------------------------------------------------------------------------------------------------------------------

## Model Testing

We can deploy the following pipeline in AWS sagemaker to track tweets real-time.

```
# Create the sequences
sample_sequences = tokenizer.texts_to_sequences(data)
sample_padded = pad_sequences(sample_sequences, padding='post', maxlen=max_len)           

# Predict the class
Prediction = sentiment[np.around(model6.predict(sample_padded), decimals=0).argmax(axis=1)[0]]
print('\n', 'Prediction =', Prediction)
```
