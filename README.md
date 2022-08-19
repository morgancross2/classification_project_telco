# Telco Churn Classification Project
by: Morgan Cross

### Project Description:
This project is designed to identify key drivers of customer churn and develop a model to best predict future customer churn. 

-----
### Project Overview:

#### Objectives:
- Document code, process (data acquistion, preparation, exploratory data analysis and statistical testing, modeling, and model evaluation), findings, and key takeaways in a Jupyter Notebook Final Report.
- Create modules (acquire.py, prepare.py) that make the process repeateable and the report (notebook) easier to read and follow.
- Ask exploratory questions of the data that will help you understand more about the attributes and drivers of customers churning. Answer questions through charts and statistical tests.
- Construct a model to predict customer churn using classification techniques, and make predictions for a group of customers.
- Refine work into a Report, in the form of a jupyter notebook, that you will walk through in a 5 minute presentation to a group of collegues and managers about the work you did, why, goals, what you found, your methdologies, and your conclusions.
- Be prepared to answer panel questions about your code, process, findings and key takeaways, and model.

#### Business Goals:
- Find drivers for customer churn at Telco. Why are customers churning?
- Construct a ML classification model that accurately predicts customer churn.
- Deliver a report that a non-data scientist can read through and understand what steps were taken, why and what was the outcome?

#### Audience:
- Peers, direct manager, and their manager

#### Project Deliverables:
- this README.md walking through the project details
- final_report.ipynb displaying the process, findings, models, key takeaways, recommendation and conclusion
- acquire.py with all data acquisition functions used
- prepare.py with all data prepareation functions used
- predictions.csv with the best models probability of churn and prediction of churn for each customer in the test dataset
- working_report.ipynb showing all work throughout the pipeline

### Data Dictionary:
| Target | Type | Description |
| ---- | ---- | ---- |
| churn | int | 0 if the customer is still with the company, 1 if they have left/churned |

| Feature Name | Type | Description |
| ---- | ---- | ---- |
| contract_type | int | 12 for month-to-month contract, 1 for 1 year contract, 2 for 2 year contract |
| customer_id | object | individual customer identifier |
| DSL | int | 0 if the customer does not have DSL, 1 if they do |
| extras | int | count of add-on services the customer is subscribed to (online security, online backup, device protection, tech support, streaming tv, streaming movies) | 
| Fiber optic | int | 0 if the customer does not have fiber optic, 1 if they do |
| monthly_charges | float | price of monthly services charged to the customer each month |
| paperless_billing | int | 0 if customer does not have paperless billing, 1 if they do |
| senior_citizen | int | 0 for non-senior citizen customers, 1 for senior citizens |
| tenure | int  | years customer has been with telco |

### Hypothesis:

1. 
- Ho -> The mean of monthly charges for churned customers is less than or equal to the mean of customers that have not churned
- Ha -> The mean of monthly charges for churned customers greater than the mean of customers that have not churned
- Outcome: I rejected the Null Hypothesis, suggesting the mean of monthly charges for churned customers is greater than those that have not churned.

2. 
- Ho2 -> There is no association between a customer having fiber optic and a customer churning
- Ha2 -> There is an association between a customer having fiber optic and a customer churning
- Outcome -> I rejected the Null Hypothesis, suggesting there is an association between a customer having fiber optic and churning.

3. 
- Ho3 -> There is not an association between tenure and churn
- Ha3 -> There is an association between tenure and churn
- Outcome -> I rejected the Null Hypothesis, suggesting there is an association between tenure and churn.

-----
## Executive Summary:
- The classification models created (Logistic Regression, Random Forest, and KNeighbors) produced varying levels of accuracy and recall. Every model beat the baseline accuracy and would better predict churn. 
- Due to wanting to capture every possible customer churning, the best model is the model that maximizes recall. The Logistic Regression model produced the best results at 79% recall.
- With more time, I predict feature engineering data to show time from subscribing to fiber optic to time of churn would increase the model's recall rate.
- I recommend evaluating fiber optic customers' experience early and often. My best model evaluated the fiber optic feature to have a weight of 2.78 (the closer to 1, the less impact on churn), over twice the next highest coefficient in the model's decision function.

-----
## Data Dictionary
| Target | Type | Description |
| ---- | ---- | ---- |
| churn | int | 0 if the customer is still with the company, 1 if they have left/churned |

| Feature Name | Type | Description |
| ---- | ---- | ---- |
| contract_type | int | 12 for month-to-month contract, 1 for 1 year contract, 2 for 2 year contract |
| customer_id | object | individual customer identifier |
| DSL | int | 0 if the customer does not have DSL, 1 if they do |
| extras | int | count of add-on services the customer is subscribed to (online security, online backup, device protection, tech support, streaming tv, streaming movies) | 
| Fiber optic | int | 0 if the customer does not have fiber optic, 1 if they do |
| monthly_charges | float | price of monthly services charged to the customer each month |
| paperless_billing | int | 0 if customer does not have paperless billing, 1 if they do |
| senior_citizen | int | 0 for non-senior citizen customers, 1 for senior citizens |
| tenure | int  | years customer has been with telco |


-----
## Planning
 - Create README, final_report.ipynb, working_report.ipynb
 - Bring over functional acquire.py, prepare.py, explore.py, and model.py files
 - Acquire the data from the Code Up database via the acquire.py function
 - Prepare and split the data via the prepare.py functions
 - Explore the data and define hypothesis. Run the appropriate statistical tests in order to accept or reject each null hypothesis. Document findings and takeaways.
 - Create a baseline model in predicting churn and document the accuracy.
 - Fit and train three (3) classification models to predict churn on the train dataset.
 - Evaluate the models by comparing the train and validation data.
 - Select the best model and evaluate it on the train data.
 - Create a predictions csv for each observation in the test dataset using the best model's predictions. The csv should include the customer id, probability of staying, probability of churning, and prediction of churn.
 - Develop and document all findings, takeaways, recommendations and next steps. 

-----
## Data Aquisition and Preparation
Files used:
 - acquire.py
 - prepare.py

Steps taken:
 - I utilized my get_telco_data function from my acquire.py file. This function gathers the data from the Telco schema in the CodeUp database via an SQL query.
 - I utilized my prep_telco function from my prepare.py file. This function:
    - handles nulls (there were none)
    - drops the foreign id columns
    - casts monetary columns to floats
    - enumerates binomial features to 0 (false) and 1 (true) for ease of exploration and modeling (See Data Dictionary above for details.)
 - I feature engineered a column named 'extras'. This column contains a count of all add-on subscriptions customers may add. The amount of extras may play a role in churn. More to be explored in the next step. 
 - Before moving to exploration, I split the data into train (60%), validatev(20%), and test (20%) datasets; these were stratified for the target: churn.

-----
## Data Exploration
Files used:
- explore.py

Takeaways from exploration:
- Increased monthly charges, having fiber optic, and early in tenure all lead to higher rates of churn. 
- A disporportionatly high number of customers that churn, have fiber optic when compared to customers that do not churn. 
- When controling for tenure, this theme continues. More fiber optic customers churn than non-fiber optic customers. 

-----
## Statistical Analysis

### Test 1: T-Test - Churned monthly charges vs Non-churned monthly charges
- A T-Test evaluates if there is a difference in the means of two continuous variables. This test is looking at a two samples and one tail.
- This test returns a p-value and a t-statistic.
- This test will compare the monthly charges of customers that have churned against the monthly charges of customers that have not churned.

Hypothesis:
 - The null hypothesis is the mean of monthly charges for churned customers is less than or equal to the mean of customers that have not churned.
 - The alternate hypothesis is the mean of monthly charges for churned customers greater than the mean of customers that have not churned.

Confidence level: 95% -> Alpha = 0.05

Results: 
- p-value is less than alpha
- t-statistic is positive

Findings: 
- I rejected the Null Hypothesis, suggesting the mean of monthly charges for churned customers is greater than those that have not churned.

### Test 2: Chi-Square - Fiber optic vs Churn
- This test evaluates if there is an association between two categorical variables.
- This test returns a chi2-value, a p-value, the degrees of freedom, and the expected outcome.
- This test will compare the fiber optic feature and the churn feature.

Hypothesis:
- The null hypothesis is there is no association between a customer having fiber optic and a customer churning.
- The alternative hypothesis is there is an association between a customer having fiber optic and a customer churning.

Confidence level: 95% -> Alpha = 0.05

Results: 
- p-value is less than alpha

Findings: 
- I rejected the Null Hypothesis, suggesting there is an association between a customer having fiber optic and churning.

### Test 3: Chi-Square - Tenure vs Churn
- This test evaluates if there is an association between two categorical variables.
- This test returns a chi2-value, a p-value, the degrees of freedom, and the expected outcome.
- This test will compare the tenure feature and the churn feature.

Hypothesis:
- The null hypothesis is there is not an association between tenure and churn.
- The alternative hypothesis is there is an association between tenure and churn.

Confidence level: 95% -> Alpha = 0.05

Results: 
- p-value is less than alpha

Findings: 
- I rejected the Null Hypothesis, suggesting there is an association between tenure and churn.

-----
## Modeling:
### Model Preparation:

### Baseline:
Baseline Results
- Train churn feature's mode is 0, not churning.
- The baseline accuracy is 73.47%.

Selected features to input into models:
- DSL
- monthly_charges
- paperless_billing
- senior_citizen
- tenure
- extras

### Model 1: Logistic Regression
Hyperparameters:
- C = 1.0
- frequency = 0.3
- random_state = 123

### Model 2: Random Forest
Hyperparameters:
- max_depth = 7
- min_samples_leaf = 3
- random_state = 123

### Model 3: K-Nearest Neighbors
Hyperparameters:
- n_neighbors = 5
- weights = uniform

### Selecting the Best Model:
| Model | Train Accuracy | Validate Accuracy | Train Recall | Validate Recall |
| ---- | ----| ---- | ---- | ---- |
| Baseline | .734675 | n/a | .734564 | n/a |
| Logistic Regression | 0.752426 | 0.765791	| 0.742194 | 0.783422 |
| Random Forest | 0.817278 | 0.808375 | 0.533452 | 0.516043 |
| K-Nearest Neighbors | 0.840237 | 0.776437 | 0.624442 | 0.524064 | 

The Logistic Regression model performed the best for recall.

### Testing the Model:
- Test Accuracy: 0.759404
- Test Recall: 0.778075

-----
## Conclusion:
It makes sense that there is more churn in the early years of tenure as customers find services that best suit their situation or take advantage of a new customer deal. However, despite this early spike in churn, fiber optic customers consistently churn at a higher rate throughout all tenure lengths. My best model evaluated the fiber optic feature to have a weight of 2.78 (the closer to 1, the less impact on churn), over twice the next highest coefficient in the model's decision function. It is this feature that should be addressed. 

#### Recommendations: 
 - Evaluate fiber optic customers' experience early and often. Themes in this customer feeback could point the company in a direction to better deter churn. 
 - Add data or begin tracking customers' location of service. Customers may be able to sign up even if fiber optic is not available in their area. Service outages causing churn may be geographically clustered. 

#### Next Steps:
 - Feature engineer sample populations where the cluster of churn is at for a collection of features. I would first isolate customers with low-tenure and high monthly charges and see how this additional identifier adjusted the model outcomes. 
 - Feature engineer data to show the last added on service before churn and the difference in dates between the addition and churning. This information could shed light on a specific service lowering customer satisfaction.

-----
## How to Recreate:
1. Utilize the following files found in this repository:
- final_report.ipynb
- acquire.py
- prepare.py
- explore.py
- model.py

2. To access the correct MySQL database, you will need credentials to access to the CodeUp database.
Create your personal env.py file with your credentials saved as strings. (user, password, host)

3. Run the final_report.ipynb notebook.