# Credit Risk Classification

![Risk](/Credit_Risk/Resources/image.png?raw=true "Risk")

In this repository, various techniques are used to train and evaluate a model based on loan risk. A dataset of historical lending activity from a peer-to-peer lending services company was used to build a model that can identify the creditworthiness of borrowers.

# Credit Risk Report

## Overview of the Analysis

The purpose of this analysis is to predict whether or not a loan is considered high risk or healthy.

Using a supervised machine learning model, specifically a Logistic Regression model, we are able to assess the data to determine the creditworthiness of borrowers.

The data used is is csv file containing 77536 records with information pertaining to the loan, including:
- `loan_size`
- `interest_rate`
- `borrower_income`
- `debt_to_income`
- `num_of_accounts`
- `derogatory_marks`
- `total_debt`
- `loan_status`

The dependent variable (or y value) was the `loan_status` column; of these records, 75036 had the label `0` (healthy), and 2500 had the label `1` (high risk).

The independent variables (or X values) were the remaining columns listed above.

Using the original data, the `y` variable and `x` variables were defined, then the data was split into training and testing sets using `train_test_split` from the scikit-learn library.

Using the training set, a `LogisticRegression` model was fitted with a random state of 1.

This model was then used to make predictions on the testing set, with results stored in a Pandas dataframe.

Finally, a confusion matrix and classification report was generated, giving valuable insights into the effectiveness of the model and therefore the accuracy of the predictions.


## Results

![Results](/Credit_Risk/Resources/results.jpg?raw=true "Results")

When looking at the model overall, it is able to predict both the `0` and `1` labels well with a 99% accuracy score, which is excellent. 

However, this changes significantly when analysing each label separately:

- The F1 score for the `0` label is 100%, because the precision score and recall score are also 100%.

- The F1 score for the `1` label is 88%, because the precision score is 87% and the recall score is 89%. This is likely due to the fact there are significantly fewer records for the `1` label (2,500) than the `0` label (75036).


## Summary

This logistic regression model proves to be an excellent tool to predict the accuracy of whether a loan can be classed as healthy or high-risk for any given borrower.

However, given the imbalance of data for the two types of loans, the model may not be appropriate to predict high-risk loans in particular, until significantly greater datapoints are added to the dataset.

Furthermore, this model only returns a black and white, or binary, response; there is not enough detail in the results to determine if a borrower may be classed as 'high-risk', but is placed on the lower end of the 'high-risk' spectrum, for example. 
