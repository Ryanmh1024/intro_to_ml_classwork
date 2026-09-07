# Final Analytics Project

Our project utilizes a Device Addiction dataset to predict whether a person will be addicted to their device or not.

[Link to Dataset on Kaggle](https://www.kaggle.com/datasets/jayjoshi37/smartphone-usage-and-addiction-prediction)

## Background info

The data set that we will be examining is on cellphone addiction. The main question we want to answer is: **"Given information about a person and their cell phone usage can we predict if they are addicted to their cell phone?"** The data set has 7500 rows with different features including age, gender, screen time, stress levels, etc. It also has a category which specifies if the person is addicted or not. We use this as the true classes that we compare our predicted classes to. This dataset is a simulation of real-world behavioral data and consists of synthetic user records which are designed for predicting digital addiction labels.

We decided on this data set because it highlights a large device addiction problem in society, specifically for younger adults. By building and evaluating models on this data set, we will be able to determine the specific metrics that contribute most to being addicted to cell phones. By discovering the specific metrics that contribute most to addiction, we are able to help prevent cellphone addiction by warning people on the biggest causes of addiction, thus enabling them to do their best to avoid cellphone addiction themselves.

## Model Evaluation

### Decision Tree Model
The final decision tree model achieved a balanced accuracy of 95.85%, an accuracy of 94.13%, and a recall of 91.71%. Since false negative is the number that we are trying to reduce, the recall is looking good and and the model is performing well comparing to the prevalence of 70.77%. This if further reflected by the AUC score of 0.9894. It is worth noting the the model actually has a precision of 1, so there were no identified cases of false positives. In relation to the problem, two variables dominates in importance, gaming hour and work study hour, which came rather unexpected, but offers valuable knowledge towards answering our question: given the gaming and work study hours, we can reliably determine whether someone is addicted.

### Random Forest Model
After building and testing, the model was evaluated using the accuracy, cross-validation accuracy, a confusion matrix, the ROC curve, and the classification report (to examine precision and recall). The accuracy proved to be quite high at 0.933. Similarly, the cross-validation accuracy (with 5 folds) was consistent with the accuracy `[0.93 ,0.929, 0.934, 0.932, 0.927]`. The confusion matrix only showed a total of 125 errors out of 1875 instances, with 66 false negatives and 59 false positives. The ROC curve was interesting, as it seemed to be nearly along the x and y axes. It had a 0.99 AUC. Lastly, the classification report showed the precisions and recalls. These were relatively high, but the model seemed to be able to identify a 1 (0.96 precision, 0.95 recall) better than a 0 (0.88 precision, 0.89 recall). As they relate to the question, the evaluation statistics show that the Random Forest model is quite effective. The accuracy and cross validations show a high score. The only caveat to this model is how the precisions and recalls are not the same for the results. The model is able to determine an addiction better than a non-addiction. 

### Logistic Regression Model
Using the Logistic Regression Model on the test data yielded an AUC of 95.47% and Recall of 94.22%. The main features that contributed to predicting whether a person is addicted or not were daily_screen_time_hours and social_media_hours. They had feature importances of 9.35 and 7.74 respectively, whereas all other feature importances had importance values below 1. The AUC of 95.47% shows that the Logistic Regression Model is able to predict very well, with mostly true positives as opposed to false positives. This is especially impressive because our prevalence is 70.77%, so this model is a significant improvement over merely guessing. Recall is made of TP/TP+FN, so with a Recall score of 94.22%, that means this model has done a good job of reducing false negatives.

## Conclusions
Since the AUC scores for all three models were quite strong, we can be confident that the models can predict whether a person will be addicted to their cellphone, as we asked in our project question. The primary indicators for cellphone addiction that we discovered are high hours of gaming, low hours of work/studying, high daily screen time hours and high daily social media hours. If some or all of these are true for an individual, it is almost certain that they are addicted to their cellphone. Given the results of the data, we would recommend that preventing cellphone addiction is best practiced by limiting high gaming time, daily screen time, and social media time, and possibly increasing the number of hours spent on work/studying. 

## Team Contribution
Jason Chen: EDA, Decision Tree model construction, Decision Tree model evaluation.

Ryan Healy: Background info, Logistic Regression model construction, Logistic Regression model evaluation, Conclusion.

Luca Sambat: Background info, Random Forest model construction, Random Forest model evaluation.
