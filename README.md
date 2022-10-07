# Mini-project IV

### [Assignment](assignment.md)

## Project/Goals
The goal of this project is to predict the approval of a loan by designing a classification machine learning model. The target variable used was 'Loan_Status', as it represents whether a loan was approved or not.

## Hypothesis
I hypothesize that those with high income and low number of dependents will have their loan approved, as these features infer relatively higher financial stability. These hypotheses will be tested by checking their correlation to the target variable, loan status.

## EDA 
- Applicannt income has the highest correlation to loan status out of all numerical variables.
- In general, those with fewer number of dependents have a higher frequency of loan approvals.
- Those with an income of 30,000 or higher all have a graduate level of education.
- Applicant Income average is at around 5,400, but with the median at around 3,800, the distribution is quite skewed due to outliers with very high income.
- The same applies to coapplicant income.
- Loan amount is also skewed, with outliers having very large loans.

## Process
#### Step 1: EDA 
- analysis of variables to determine relationships and their properties
#### Step 2: Cleaning
- null value replacements:
    - gender: filled with Male (most frequent value)
    - marital status: filled with yes (most frequent value)
    - dependents: filled with 0 (most frequent value)
    - self-employed: filled with no (most frequent value)
    - loan amount: filled with mean value
    - loan term: filled with 360.0 (most frequent value)
    - credit history: filled with 1.0 (most frequent value)
#### Step 3: Feature Engineering 
- Total income feature created by combining applicant and coapplicant income.
- Log transformations on loan amount and total income.
#### Step 4: Modeling
- Combination of PCA and SelectKBest for feature selection.
- FeatureUnion and Pipeline to build model.
- Gradient Boosting Classifier used for ML algorithm.
#### Step 5: Deployment
- API creation
- Pickling of trained model
- Uploaded to AWS instance
#### Step 6: Testing
- Tested API and model via Python

## Results/Demo
Although I did not have enough time to check the performance of the model via accuracy/precision/recall/f1 scores, the model successfully predicted a randomly selected row taken out of the dataset prior to training the model.
The API initially had a few issues, but were resolved by matching the scikit-learn package version to the one used in the process of modeling. After the fixes, the API is now working seamlessly.

## Challanges 
Lack of time was the biggest challenge in this project. I had to choose a classification model without checking and picking the model with the best performance, and did not have time to thoroughly evaluate the model performance. Additionally, version control with the deployment onto AWS was also tricky and used up a lot of my time.

## Future Goals
Further investigation on the performance of other classification models is recommended, as well as evaluating the model's performance by performing other evaluation metrics. If possible, a larger sample for the data would be constructive.

## Notes
The project was completed with all its files in one folder - to run the project without errors, place the following files in the same folder:
- instructions.ipynb (stored in notebooks folder)
- app.py (stored in src folder)
- loan_model.sav (stored in src folder)
- data.csv (stored in data folder)