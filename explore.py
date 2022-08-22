import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

def explore_cats(train, cats, target):
    '''
    This function takes:
            train = train DataFrame
            cats = category columns (as a list of strings)
            target = target variable (as a string)
    prints value counts for each category in each column
    '''
    for col in cats:
        print(col)
        print(train[col].value_counts())
        print(train[col].value_counts(normalize=True)*100)
        sns.countplot(x=col, data=train)
        plt.title(col+' counts')
        plt.show()
    
        sns.barplot(data=train, x=col, y=target)
        rate = train[target].mean()
        plt.axhline(rate, label= 'average ' + target + ' rate')
        plt.legend()
        plt.title(target+' rate by '+col)
        plt.show()
    
        alpha = 0.05
        o = pd.crosstab(train[col], train[target])
        chi2, p, dof, e = stats.chi2_contingency(o)
        result = p < alpha
        print('P is less than alpha: '+result.astype('str'))
        print('------------------------------------------------------------')

def explore_nums(train, nums):
    '''
    This function takes in:
            train = train DataFrame
            nums = numerical columns (as a list of strings)
    '''
    for col in nums:
        sns.histplot(x=col, data=train)
        plt.show()
        
def plot_chi(train, cats):
    alpha = 0.05
    for col in cats:
        o = pd.crosstab(train[col], train[target])
        chi2, p, dof, e = stats.chi2_contingency(o)
        plt.plot(train[col], chi2)
    plt.show()    
    
def show_monthly_charges(train):
    ''' This function takes in the train dataset and outputs the visualization for question 1 
    in the telco churn project addressing monthly charges and churn.
    '''
    
    # Create the plot frame
    fig, axes = plt.subplots(2,1, figsize=(16,12))

    # Give it a title
    fig.suptitle('''Churning Customers Experience $70+ Monthly Charges More Often''')

    # Plot it as a histograph with distribution line
    sns.histplot(ax=axes[0], data=train[train.churn == 0], x='monthly_charges')
    sns.histplot(ax=axes[0], data=train[train.churn == 1], x='monthly_charges', color='sandybrown')
    # Put a red box around where we will be zooming into
    axes[0].hlines(300,15,120, colors='red')
    axes[0].hlines(.2,15,120, colors='red')
    axes[0].vlines(15,0,300, colors='red')
    axes[0].vlines(120,0,300, colors='red')

    # Create lines showing average monthly charge for each group, then label them
    avg_churn_cost = train[train.churn==1].monthly_charges.mean()
    avg_non_churn_cost = train[train.churn==0].monthly_charges.mean()
    axes[0].axvline(avg_churn_cost, label= 'Average Churn Monthly Charge', color='saddlebrown', linestyle='--')
    axes[0].annotate('Avg Churn Charge', xy=(75,500), xytext = (80, 500), arrowprops={'arrowstyle':'->', 'color':'black'})
    axes[0].axvline(avg_non_churn_cost, label= 'Average Non-Churn Monthly Charge', color='midnightblue', linestyle='--')
    axes[0].annotate('Avg Non-Churn Charge', xy=(62,600), xytext = (35, 600), arrowprops={'arrowstyle':'->', 'color':'black'})


    # Plot the zoomed in graph
    sns.histplot(ax=axes[1], data=train[train.churn == 0], x='monthly_charges')
    sns.histplot(ax=axes[1], data=train[train.churn == 1], x='monthly_charges', color='sandybrown')
    axes[1].set_ylim(0,300)
    axes[1].set_xlim(15,120)

    # Create lines showing average monthly charge for each group, then label them
    avg_churn_cost = train[train.churn==1].monthly_charges.mean()
    avg_non_churn_cost = train[train.churn==0].monthly_charges.mean()
    axes[1].axvline(avg_churn_cost, label= 'Average Churn Monthly Charge', color='saddlebrown', linestyle='--')
    axes[1].annotate('Avg Churn Charge', xy=(75,283), xytext = (80, 283), arrowprops={'arrowstyle':'->', 'color':'black'})
    axes[1].axvline(avg_non_churn_cost, label= 'Average Non-Churn Monthly Charge', color='midnightblue', linestyle='--')
    axes[1].annotate('Avg Non-Churn Charge', xy=(62,260), xytext = (40, 260), arrowprops={'arrowstyle':'->', 'color':'black'})

    # Give the individual graphs titles
    axes[0].set_title("Distribution of Customer's Monthly Charges")
    axes[1].set_title("Closer Look at Spike in Churned Customers Monthly Charges")

    plt.show()

    
def show_fiber_optic(train):
    ''' This function takes in the train dataset and outputs the visualization for question 2 
    in the telco churn project addressing fiber optic and churn.
    '''
    # plot the count of customers in the train dataset against the fiber optic feature
    plt.figure(figsize=(12,6))
    sns.countplot(x='Fiber optic', data=train)
    plt.title('Fiber Optic Customer Counts')
    plt.show()

    # calculate the count of customers with and without fiber optic
    no_fo = train[['Fiber optic']].value_counts()[0]
    yes_fo = train[['Fiber optic']].value_counts()[1]

    # calculate the percentage of customers with and without fiber optic
    no_fo_per = round((train[['Fiber optic']].value_counts(normalize=True)*100)[0], 1)
    yes_fo_per = round((train[['Fiber optic']].value_counts(normalize=True)*100)[1], 1)

    # print the previously calculated values in statements
    print(f'{no_fo} customers or {no_fo_per}% do not have fiber optic.')
    print(f'{yes_fo} customers or {yes_fo_per}% have fiber optic.')

    # make some space between graphs
    print('')
    print('---------------------------------------------------------------------------------------------------------------')
    print('')

    # plot the feature fiber optic against the target of churn
    plt.figure(figsize=(12,6))
    sns.barplot(data=train, x='Fiber optic', y='churn')

    plt.title('Churn Rate for Fiber Optic Customers')
    plt.show()

    # calculate the churn rate for customers with and without fiber optic
    yes_churn_fo_per = round(((train[(train['Fiber optic']==1)&(train.churn)==1].shape[0]) / (train[train['Fiber optic'] == 1].shape[0]))*100, 1)
    no_churn_fo_per = round(((train[(train['Fiber optic']==0)&(train.churn)==1].shape[0]) / (train[train['Fiber optic'] == 0].shape[0]))*100, 1)

    # print the previous calulations in statements
    print(f'{no_churn_fo_per}% of all customers without fiber optic churn.')
    print(f'{yes_churn_fo_per}% of all customers with fiber optic churn.')
   



def show_tenure(train):
    ''' This function takes in the train dataset and outputs the visualization for question 3 
    in the telco churn project addressing tenure and churn.
    '''
    # Make the plot larger
    plt.figure(figsize=(12,6))
    # Plot it as a histogram
    sns.histplot(data=train, x='tenure', bins=72)
    # Give it a title
    plt.title("Distribution of all cutomers' tenure")
    plt.show()
    
    # Make the plot larger
    plt.figure(figsize=(12,6))
    # Plot it as a histogram with a distribution line, create one bar per year
    sns.histplot(data=train[train.churn == 1], x='tenure', bins=72)
    plt.xlim(0,72)
    # Create a box on where we are going in the next graph
    plt.hlines(100,0,20, colors='red')
    plt.hlines(.2,0,20, colors='red')
    plt.vlines(.2,0,100, colors='red')
    plt.vlines(20,0,100, colors='red')
    # Give it a title
    plt.title("Distribution of churning customers' tenure shows spike in early years")
    plt.show()
    
    print("These two histographs show that customers churn the most early on in tenure. This makes sense with new customer deals expiring. There seems to be another drop off. Let's zoom in and look at the first 20 years of churning customers")
    print()
    
    # Make the plot larger
    plt.figure(figsize=(12,6))
    # Plot it as a histogram with a distribution line, create one bar per year
    sns.histplot(data=train[train.churn == 1], x='tenure', bins=72)
    # Zoom into the box from the previous histogram
    plt.xlim(0,20)
    plt.ylim(0,100)
    # Give it a title
    plt.title("Distribution of churning customers' tenure shows spike in early years")
    plt.show()
    
    
def show_combo(train):
    ''' This function takes in the train dataset and outputs the visualization for putting 
    all the explore results together in the telco churn project addressing monthly charges,
    fiber optic, tenure, and churn.
    '''
    # Create subplot framework
    fig, axes = plt.subplots(2,2, figsize=(12,12))
    # Give the whole thing a title
    fig.suptitle('''The Rate of Churn is Disproportionally High for Fiber Optic Customers: 
    Displayed Controlled for Tenure''')

    # Give titles to each smaller graph
    axes[0,0].set_title('Churned Customers vs Fiber Optic')
    axes[0,1].set_title('Churned Customers Over Tenure vs Fiber Optic')
    axes[1,0].set_title('Kept Customers vs Fiber Optic')
    axes[1,1].set_title('Kept Customers Over Tenure vs Fiber Optic')

    # Plot each quadrant't graph and give it a location to populate
    sns.histplot(ax=axes[0,0], data=train[train.churn == 1], x='monthly_charges', hue='Fiber optic', kde=True)
    axes[0,0].set_ylim(0,250)
    sns.scatterplot(ax=axes[0,1], data=train[(train.churn == 1)], y='tenure', x='monthly_charges', hue='Fiber optic')
    sns.histplot(ax=axes[1,0], data=train[train.churn == 0], x='monthly_charges', hue='Fiber optic', kde=True)
    axes[1,0].set_ylim(0,250)
    sns.scatterplot(ax=axes[1,1], data=train[(train.churn == 0)], y='tenure', x='monthly_charges', hue='Fiber optic')
    plt.show()