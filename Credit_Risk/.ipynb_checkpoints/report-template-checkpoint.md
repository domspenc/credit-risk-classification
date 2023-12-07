# Module 12 Report Template

## Overview of the Analysis

The purpose of this analysis is to predict whether or not a loan is considered high risk or healthy.

Using a supervised machine learning model, specifically a Logistic Regression model, we are able to assess the data to determine the creditworthiness of borrowers.

The data used is is csv file containing 77536 records with information pertaining to the loan, including `loan_size`, `interest_rate`, `borrower_income`, `debt_to_income`, `num_of_accounts`, `derogatory_marks`, `total_debt` and `loan_status`. 

The dependent variable (or y value) was the `loan_status` column; of these records, 75036 had the label `0` (healthy), and 2500 had the label `1` (high risk).

The independent variables (or X values) were the remaining columns listed above.

Using the original data, the `y` variable and `x` variables were defined, then the data was split into training and testing sets using `train_test_split` from the scikit-learn library.

Using the training set, a `LogisticRegression` model was fitted with a random state of 1.

This model was then used to make predictions on the testing set, with results stored in a Pandas dataframe.

Finally, a confusion matrix and classification report was generated, giving valuable insights into the effectiveness of the model and therefore the accuracy of the predictions.


## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Results:
  * Description of Model 1 Accuracy, Precision, and Recall scores.


## Summary

Summarise the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.
