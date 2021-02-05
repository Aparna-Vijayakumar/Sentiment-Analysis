# Sentiment-Analysis

### Overview:
This project uses a simple machine learning model to classify the sentiment of each product review.

### Dataset:
This project uses the Amazon Review dataset, which can be found at https://nijianmo.github.io/amazon/index.html <br>
Rather than using the entire dataset, I used only the Beauty, Food, and Automotive categories for my analysis.

### Preprocessing:
The data in each of the above categories was preprocessed before feeding into the model as input. Preprocessing was done using the NLTK package and involves the following steps: 
<li>Parsing to retain only the text between the review text tags and the rating between rating tags </li>
<li> Tokenization </li>
<li> Removing punctuation </li>
<li> Removing stopwords </li>
<li> Conversion of all tokens to lowercase </li>
<li> Reducing all tokens to their rootwords (Stemming) </li>
Here the review text is what we will be classifying, and the rating will be our label for training.

###  Converting Ratings to Labels for Sentiment Analysis:
Looping through the dataframe, everywhere the rating was greater than 3 we change it to 1
(denoting a positive review), if the rating was less than 3 we change it to -1 (denoting a negative
review), and if the rating was equal to 3 we change it to 0 (denoting a neutral review). Since we
discard all reviews with a rating 3, we then drop all rows from my dataframe which have a label
of 0.

###  Obtain Word Counts:
We convert the data into a dataframe of counts such that each column corresponds to
each word that appears across all reviews, and each row corresponds to the count of each
word. This is the final data that we train our model on. We then write this dataframe of word counts onto a CSV file which represents the training
data, and the converted ratings onto another CSV file, which represents the labels.

### Modelling:
We use a simple Logistic Regression model, since the problem task is a binary classification task
where 1 represents a positive review and 0 represents a negative review. WE first read the CSV file containing the feature representation that we created above into a dataframe
X, and the CSV file containing the positive and negative labels into a dataframe y.
We then use the train test split function in the sklearn package in Python to split the data as
75% for training and 25% for testing.
For each value of hyperparameter C, we fit a new Logistic Regression Model and run the
model for 10 epochs.
The values we use for C are : <br>
C = 0.01, 0.05, 0.25, 0.5, 1, 1.2, 1.5, 1.8, 2 <br>
Since the data is quite sparse (with majority of zeros for counts), we use both accuracy score as
well as f1 score to evaluate the model.

### Results:
We get highest training accuracy scores with C = 2 on the beauty data, we choose that as our final model and evaluate on the test set to obtain: <br>
Test accuracy : 0.9972260748959778, Test f1 score : 0.9983498349834983

###  Interpreting the Model
To check if the model makes sense, we want to see the top 5 words that had a positive sentiment.<br>
Top 5 positive words for beauty reviews : 
<li>'home', 2.11 <\li>
<li>'easi', 1.83 <\li>
<li>'believ', 1.61 <\li>
<li>'best', 1.53 <\li>
<li>'smooth', 1.5 <\li>

###  Cross Domain Sentiment Classification
We test the model which was trained on gourmet food reviews on automotive reviews, which gives the following results : <br>
Train Data : Gourmet Food, Test Data : Automotives <br>
Test accuracy : 0.7989130434782609, Test f1 score : 0.8804523424878837 <br>
We obtained lesser scores when testing our model on a different category of reviews. This
makes sense as the two categories are mostly unrelated to each other. Delicious would be a
positive word for food, but it does not make any sense for it to appear in a an automotive review.
If we wanted the model to perform better for other category of reviews, during the preprocessing stage we could omit those words that are specific for that particular category.
For example, removing smooth from the reviews of beauty products could make the model
perform better on unrelated categories like tools and hardware.




