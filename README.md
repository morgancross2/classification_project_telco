# Telco Churn Classification Project
by: Morgan Cross

This project is designed to identify key drivers of customer churn and develop a model to best predict future customer churn. The results from this project arms telco business leaders and decision makers with a focused direction to best address these drivers and retain customers. 


A Canva Presentation of this material can be found [here.](https://www.canva.com/design/DAFKRW7Uez4/E44XUzojHue9Q9n2ds76Xw/view?utm_content=DAFKRW7Uez4&utm_campaign=designshare&utm_medium=link&utm_source=recording_view)
For just the slides, use [this one!](https://www.canva.com/design/DAFKRW7Uez4/6MFGLWBLbQqxpPV0AzdKUQ/view?utm_content=DAFKRW7Uez4&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

-----
## Project Overview:

#### Objectives:
- Document code, process (data acquistion, preparation, exploratory data analysis and statistical testing, modeling, and model evaluation), findings, and key takeaways in a Jupyter Notebook Final Report.
- Create modules (acquire.py, prepare.py) that make the process repeateable and the report (notebook) easier to read and follow.
- Ask exploratory questions of the data that will help you understand more about the attributes and drivers of customers churning. Answer questions through charts and statistical tests.
- Construct a model to predict customer churn using classification techniques, and make predictions for a group of customers.
- Refine work into a Report, in the form of a jupyter notebook, that you will walk through in a 5 minute presentation to a group of collegues and managers about the work you did, why, goals, what you found, your methdologies, and your conclusions.
- Be prepared to answer panel questions about your code, process, findings and key takeaways, and model.

#### Business Goals:
- Find drivers for customer churn at Telco. Why are customers churning?
- Construct a ML classification model that best predicts customer churn.
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

-----
## Executive Summary:
Goals:
- Identify drivers of churn
- Build a model to delve into the root of the churning problem
- Maximize recall in order to predict churning customers and minimize false negatives

Key Findings:
- High monthly charges, low tenure, and fiber optic are primary drivers of churn
- The classification models (Logistic Regression, Random Forest, and KNeighbors) produced varying levels of accuracy and recall.

Takeaways:
- The Logistic Regression model produced the best results at 79% recall. 
- This model is built to delve into the root of the problem, not to isolate predicted churning customers for targeted marketing. Follow on models before marketing campaigns are highly recommended. 

Recommendation:
- I recommend conducting research in possible competitors offering fiber optic. Despite churn overall declining as tenure increases, fiber optic customers are churning much more frequently at all tenures. This heavily suggests customers are leaving due to this service and it is likely it is being offered faster or for less.

-----
## Data Dictionary:
| Target | Type | Description |
| ---- | ---- | ---- |
| churn | int | 0 if the customer is still with the company, 1 if they have left/churned |

| Feature Name | Type | Description |
| ---- | ---- | ---- |
| Bank transfer (automatic) | int | 0 if the customer does not use bank transfering, 1 if they do |
| contract_type | int | 12 for month-to-month contract, 1 for 1 year contract, 2 for 2 year contract |
| contract_type_id | int | foreign key to contract_type |
| Credit card (automatic) | int | 0 if the customer does not use a credit card, 1 if they do |
| customer_id | object | individual customer identifier |
| dependents | int | 0 if the customer does not have dependents, 1 if they do |
| device_protection | int | 0 if the customer does not have device protection, 1 if they do |
| DSL | int | 0 if the customer does not have DSL, 1 if they do |
| Electronic check | int | 0 if the customer does not use electronic checks, 1 if they do |
| extras | int | count of add-on services the customer is subscribed to (online security, online backup, device protection, tech support, streaming tv, streaming movies) | 
| Fiber optic | int | 0 if the customer does not have fiber optic, 1 if they do |
| gender | int | 0 if the customer is female, 1 if they are male |
| internet_service_type_id | int | foreign key to internet_service_type |
| Mailed check | int | 0 if the customer does not use mailed checks, 1 if they do |
| monthly_charges | float | price of monthly services charged to the customer each month |
| multiple_lines | int | 0 if the customer does not have any lines, 1 if they have one line, 2 if they have two or more lines |
| online_backup | int | 0 if the customer does not have online backup, 1 if they do |
| online_security | int | 0 if the customer does not have online security, 1 if they do |
| paperless_billing | int | 0 if customer does not have paperless billing, 1 if they do |
| partner | int | 0 if the customer does not have a partner, 1 if they do |
| payment_type | object | This feature gets broken down into: bank transfer, credit card, electronic check, and mailed check |
| payment_type_id | int | foreign key to payment_type |
| phone_service | int | 0 if the customer does not have phone service, 1 if they do |
| senior_citizen | int | 0 for non-senior citizen customers, 1 for senior citizens |
| streaming_movies | int | 0 if the customer does not have streaming movies, 1 if they do |
| streaming_tv | int | 0 if the customer does not have streaming tv, 1 if they do |
| tech_support | int | 0 if the customer does not have tech support, 1 if they do |
| tenure | int  | years customer has been with telco |
| total_charges | float | sum of all charges over the tenure of the customer |

-----
## Planning
 - Create deliverables:
     - README
     - final_report.ipynb
     - working_report.ipynb
     - predictions.csv including customer id, probability of churn, and prediction of churn
 - Bring over functional acquire.py, prepare.py, explore.py, and model.py files
 - Acquire the data from the Code Up database via the acquire.py function
 - Prepare and split the data via the prepare.py functions
 - Explore the data and define hypothesis. Run the appropriate statistical tests in order to accept or reject each null hypothesis. Document findings and takeaways.
 - Create a baseline model in predicting churn and document the accuracy.
 - Fit and train three (3) classification models to predict churn on the train dataset.
 - Evaluate the models by comparing the train and validation data.
 - Select the best model and evaluate it on the train data.
 - Develop and document all findings, takeaways, recommendations and next steps. 

-----
## Data Aquisition and Preparation
Files used:
 - acquire.py
 - prepare.py

Steps taken:
 - I utilized my get_telco_data function from my acquire.py file. This function gathers the data from the Telco schema in the CodeUp database via an SQL query.
 - In this step, I called prep_telco from my prepare.py file. This function 
    - handles nulls
    - drops the foreign id columns
    - casts monetary columns to floats
    - enumerates columns's data for ease of exploration and modeling (See Data Dictionary above or in README for details.)
 - I feature engineered a column named 'extras'. This column contains a count of all add-on subscriptions customers may add. The amount of extras may play a role in churn. More to be explored in the next step. 
 - Before moving to exploration, I split the data into train (60%), validatev(20%), and test (20%) datasets; these were stratified for the target: churn.

What happened to each feature?
- internet_service_type_id, payment_type_id, and contract_type_id were dropped because they are foreign keys to other features that were merged to the table via the initial acquire query. 
- total_charges with no entry were corrected to reflect 0. This feature was then casted into a float.
- internet_service_type and payment_type were transformed via dummies. This change displays the categorical options for each feature in their own boolean-like columns with 1 for True and 0 for False.
- contract_type, gender, partner, dependents, phone_service, paperless_billing, churn, multiple_lines, online_security, online_backup, device_protection, tech_support, streaming_tv, and streaming_movies features were transformed to display 1 for True and 0 for False in place of 'Yes' or 'No'.


-----
## Data Exploration
Files used:
- explore.py

Questions Addressed:
1. Do monthly charges have a relationship with churn?
2. Is fiber optic a driver of churn?
3. How does tenure effect churn?

### Test 1: T-Test - Churned monthly charges vs Non-churned monthly charges
- A T-Test evaluates if there is a difference in the means of two continuous variables. This test is looking at a two samples and one tail.
- This test returns a p-value and a t-statistic.
- This test will compare the monthly charges of customers that have churned against the monthly charges of customers that have not churned.
- Confidence level is 95%
- Alpha is 0.05

Hypothesis:
 - The null hypothesis is the mean of monthly charges for churned customers is less than or equal to the mean of customers that have not churned.
 - The alternate hypothesis is the mean of monthly charges for churned customers greater than the mean of customers that have not churned.

Results: 
- p-value is less than alpha
- t-statistic is positive
- I rejected the Null Hypothesis, suggesting the mean of monthly charges for churned customers is greater than those that have not churned.

### Test 2: Chi-Square - Fiber optic vs Churn
- This test evaluates if there is an association between two categorical variables.
- This test returns a chi2-value, a p-value, the degrees of freedom, and the expected outcome.
- This test will compare the fiber optic feature and the churn feature.
- Confidence level is 95%
- Alpha is 0.05

Hypothesis:
- The null hypothesis is there is no association between a customer having fiber optic and a customer churning.
- The alternative hypothesis is there is an association between a customer having fiber optic and a customer churning.

Results: 
- p-value is less than alpha
- I rejected the Null Hypothesis, suggesting there is an association between a customer having fiber optic and churning.

### Test 3: Chi-Square - Tenure vs Churn
- This test evaluates if there is an association between two categorical variables.
- This test returns a chi2-value, a p-value, the degrees of freedom, and the expected outcome.
- This test will compare the tenure feature and the churn feature.
- Confidence level is 95%
- Alpha is 0.05

Hypothesis:
- The null hypothesis is there is not an association between tenure and churn.
- The alternative hypothesis is there is an association between tenure and churn.

Results: 
- p-value is less than alpha
- I rejected the Null Hypothesis, suggesting there is an association between tenure and churn.

### Takeaways from exploration:
- Increased monthly charges, having fiber optic, and early in tenure all lead to higher rates of churn. 
- A disporportionatly high number of customers that churn, have fiber optic when compared to customers that do not churn. 
- When controling for tenure, this theme continues. More fiber optic customers churn than non-fiber optic customers. 

-----
## Modeling:
### Model Preparation:

### Baseline:
Baseline Results
- Train churn feature's mode is 0, not churning.
- The baseline accuracy is 73.47%.

Selected features to input into models:
- contract_type
- Fiber optic
- DSL
- monthly_charges
- paperless_billing
- senior_citizen
- tenure
- extras

#### Model 1: Logistic Regression
- Hyperparameters: C = 1.0, frequency = 0.3, random_state = 123

#### Model 2: Random Forest
- Hyperparameters: max_depth = 7, min_samples_leaf = 3, random_state = 123

#### Model 3: K-Nearest Neighbors
- Hyperparameters: n_neighbors = 5, weights = uniform

### Selecting the Best Model:
| Model | Train Accuracy | Validate Accuracy | Train Recall | Validate Recall |
| ---- | ----| ---- | ---- | ---- |
| Baseline | .734675 | .734564 | n/a | n/a |
| Logistic Regression | 0.752426 | 0.765791	| 0.742194 | 0.783422 |
| Random Forest | 0.817278 | 0.808375 | 0.533452 | 0.516043 |
| K-Nearest Neighbors | 0.840237 | 0.776437 | 0.624442 | 0.524064 | 

The Logistic Regression model performed the best for recall.

### Testing the Model:
| Model | Test Accuracy | Test Recall |
| ---- | ---- | ---- |
| Logistic Regression | 0.759404 | 0.778075 |

-----
## Conclusion:
Throughout customers' tenure, fiber optic customers are consistently churning at a higher rate. It is even more prevelent that this feature that should be addressed because my model is predicting churn with 78% recall and is weighing fiber optic at 2.78, almost twice the next highest feature in its decision function. 

#### Recommendations: 
 - Conduct research on other fiber optic providers. There is likely a competitor providing this service to customers faster, better, or cheaper.
 - Add data or begin tracking customers' location of service. Customers may be able to sign up even if fiber optic is not available in their area. Service outages causing churn may be geographically clustered. 
 - Evaluate fiber optic customers' experience early and often. Themes in this customer feeback could point the company in a direction to better deter churn. 

#### Next Steps:
 - Feature engineer sample populations where the cluster of churn is at for a combination of features. Example: isolate customers with low-tenure and high monthly charges, add an identifier for this group, and see how this additional identifier adjusted the model outcomes. 
 - Feature engineer data to show the last added on service before churn and the difference in dates between the addition and churning. This information could shed light on a specific service lowering customer satisfaction.
 - Develop a model focused on targeted marketing for predicted churning customers.

-----
## How to Recreate:
1. Utilize the following files found in this repository:
- final_report.ipynb
- acquire.py
- prepare.py
- explore.py
- model.py

2. To access the data, you will need credentials to access to the CodeUp database saved in an env.py file.
Create your personal env.py file with your credentials saved as strings. (user, password, host)

3. Run the final_report.ipynb notebook.