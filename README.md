# TELCO CHURN CLASSIFICATION PROJECT
by: Morgan Cross

### Project Description:
Customers at Telco are churning. This project is designed to identify key drivers of customer churn and develop a model to predict future customer churn. 

-----
### Project Planning:
#### Project Outline:
1. Data Acquisition
2. Data Preparation
3. Data Exploratory Data Analysis and Statistical Testing
4. Data Modeling
5. Model Evaluation
6. Findings and Takeaways
7. Recommendations

#### Hypothesis:
Ho1 -> There is no association between churn and a customer having fiber optic
Ha1 -> There is an association between churn and a customer having fiber optic
Ho2 -> There is no association between churn and a customer having paperless billing
Ha2 -> There is an association between churn and a customer having paperless billing
HoFinal -> There is a less than or equal probability of churn if a customer haspaperless billing and fiber optic than if they do not
HaFinal -> There is a larger probability of churn if a customer has paperlessbilling and fiber optic than if they do not

#### Target Variable:
Churn!

#### Need to haves (Deliverables):
1. Readme.md (here we are!)
2. Final Report (.ipynb)
3. Acquire & Prepare Modules (.py)
4. Predictions (.csv)
5. Any extra working notebooks

#### Nice to haves (With more time):

-----
### Key Findings:

-----
### Data Dictionary
(table with attribue, definition, and data type)
-----
### Data Aquisition and Preparation
Data was wrangled using the files listed below. All data was gathered from the Telco schema in the CodeUp database. This data was accessed via a function in the acquire.py file that accesses MySQL and runs a SQL query. The data was prepared by:

- Deleting all foreign key columns
- Casting columns containing prices into float types
- Numerating binomial features to show as 0 (false) and 1 (true)
- Create dummy features to split out features with 2+ options
- Split the data into train (60%), validate (20%), and test (20%) datasets

Files used:
    - acquire.py
    - prepare.py
-----
### Data Exploration
Files used:
- explore.py

Takeaways from exploration:

-----
### Statistical Analysis

#### Test 1: ANOVA Test
- what it allows to test
- what it returns (variables)
- wanted to compare what columns to what columns

#### Test 2: T-Test


#### Test 3: Chi-Square

#### Hypothesis:
The null hypothesis is
The alternate hypothesis is

#### Confidence level: 95%
#### Alpha: 0.05

#### Results:

#### Findings:

-----
### Modeling:
#### Model Preparation:

#### Baseline:
Baseline Results
Selected feature to input into models:
    - feature 1
    - feature 2

#### Models and R^2 Values:
Will run the following regression models:

Other indicators of model performance:
    - indicator and why it is important

##### Model 1: type
Results

##### Model 2: type
Results

### Selecting the Best Model:
| Model | Validation/Out of Sample RMSE | R<sup>2</sup> Value |
| ---- | ----| ---- |
| Baseline | 0.167366 | 2.2204 x 10<sup>-16</sup> |
| Linear Regression (OLS) | 0.166731 | 2.1433 x 10<sup>-3</sup> |  
| Tweedie Regressor (GLM) | 0.155186 | 9.4673 x 10<sup>-4</sup>|  
| Lasso Lars | 0.166731 | 2.2204 x 10<sup>-16</sup> |  
| Quadratic Regression | 0.027786 | 2.4659 x 10<sup>-3</sup> |  

Model x performed the best.

### Testing the Model:
Results results results

### Conclusion: