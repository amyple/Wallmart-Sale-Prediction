#Group 1: SALE$TER

def main():

    # Find predicted weekly sales, top and bottom 10 departments, and which week of chosen month has a holiday. User will input store ID and month they want predictions of.
    # import libraries
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import cross_val_score
    import numpy as np
    from sklearn import preprocessing
    from sklearn import utils
    from numpy import newaxis
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn import linear_model
    from sklearn import preprocessing
    from sklearn.preprocessing import PolynomialFeatures
    import matplotlib.pyplot as plt; plt.rcdefaults()
    import matplotlib.pyplot as plt
    import walmartgraphlibrary
    
    #poly regression with degree 6 gives best results
    poly = PolynomialFeatures(degree=6)
    
    # Import walmart data 
    walmart = pd.read_csv('walmart data clean.csv', index_col = 'Store')
    
    # Make all categories into numerical values with multiple dictionaries. Change IsHoliday from bool to category for 0 and 1 values.
    walmart["IsHoliday"] = walmart["IsHoliday"].astype('category')
    walmart["IsHoliday"] = walmart["IsHoliday"].cat.codes
    num_var = {"temp_class": {"Cold":1,"Comfortable":2,"Hot":3},
               "fuel_class": {"low":1,"Medium":2,"High":3},
               "unemploy_class": {"low":1,"Medium":2,"High":3},
               "cpi_class": {"low":1,"Medium":2,"High":3},
               "size_class":{"low":1,"Medium":2,"High":3},
               "Type":{"A":1,"B":2},
               "sales_class":{"Low":1,"Medium":2,"High":3,"Negative":0}}
    
    # Replace above numerical values into table
    walmart.replace(num_var, inplace=True)
    
    # Make string date in datetime date
    walmart['Date'] = pd.to_datetime(walmart['Date'])
    
    # This dictionary ill make department number values into their name
    dept_change = {'Dept': {1: 'Candy and Tobacco',
      2: 'Health and Beauty',
      3: 'Stationery',
      4: 'Paper Goods',
      5: 'Media and Gaming',
      6: 'Cameras',
      7: 'Toys',
      8: 'Pets',
      9: 'Sporting Goods',
      10: 'Automotive',
      11: 'Hardware',
      12: 'Paint',
      13: 'Household Chemicals',
      14: 'Kitchen and Dining',
      15: 'Clinics',
      16: 'Lawn and Garden',
      17: 'Home Decor',
      18: 'Seasonal',
      19: 'Crafts and Fabrics',
      20: 'Bath and Shower',
      21: 'Books and Magazines',
      22: 'Bedding',
      23: 'Menswear',
      24: 'Boyswear',
      25: 'Shoes',
      26: 'Infant Apparel',
      27: "Ladies' Socks",
      28: 'Hosiery',
      29: 'Sleepwear/Scrubs/Underwear',
      30: 'Bras and Shapewear',
      31: 'Accessories',
      32: 'Jewelry',
      33: 'Girlswear',
      34: 'Ladieswear',
      35: 'Plus Size and Maternity',
      36: "Ladies' Outwear and Swimwear",
      37: 'Auto Services',
      38: 'Prescription Pharmacy',
      39: 'N/A',
      40: 'OTC Pharmacy',
      41: 'College/Pro Apparel (Sub 23)',
      42: 'Motor Oil (Sub 10)',
      43: 'Toys (Sub 7)',
      44: 'Crafts (Sub 19)',
      45: 'Aidco (Sub 9)',
      46: 'Cosmetics',
      47: 'Jewelry (Sub 32)',
      48: 'Firearms (Sub 9)',
      49: 'Optical',
      50: 'Optical Service Income',
      51: 'Sporting Goods (Sub 9)',
      52: 'Crafts (Sub 19)',
      53: 'Cards, Books, and Magazines (Sub 3)',
      54: 'Jewelry (Sub 32)',
      55: 'Media and Gaming (Sub 5)',
      56: 'Horticulture/Live Plants',
      57: 'Toys (Sub 7)',
      58: 'Wireless Services',
      59: 'Cosmetics/Skincare (Sub 46)',
      60: 'Concept Stores and Stamps',
      61: 'N/A',
      62: 'N/A',
      63: 'N/A',
      64: 'N/A',
      65: 'Gas',
      66: "Sam's Club",
      67: 'Celebrations',
      68: 'N/A',
      69: 'Walmart.com',
      70: "Sam's Club",
      71: 'Furniture',
      72: 'Electronics',
      73: 'Books and Magazines (Sub 21)',
      74: 'Home Management and Luggage',
      75: "Doctor's Fees",
      76: 'Academy (non-retail)',
      77: 'Large Appliances (defunct)',
      78: 'Ladieswear (Sub 34)',
      79: 'Infant Consumables and Hardlines',
      80: 'Service Deli',
      81: 'Commercial Bread',
      82: 'Impulse Merchandise and Batteries',
      83: 'Seafood (defunct)',
      84: 'Flowers and Balloons (defunct)',
      85: 'Photo Lab',
      86: 'Financial Services',
      87: 'Wireless',
      88: 'PMDC Signage (non-retail)',
      89: 'Travel Center',
      90: 'Dairy',
      91: 'Frozen Food',
      92: 'Dry Grocery',
      93: 'Fresh/Frozen Meat and Seafood',
      94: 'Produce',
      95: 'DSD Grocery, Snacks, and Beverages',
      96: 'Liquor',
      97: 'Packaged Deli',
      98: 'Bakery',
      99: 'Store Supplies (non-retail)'}}
    
    # Ask user for StoreID and Month number
    print("*****Welcome to SALE$TER!*****")
    print("Find your SALE$CAST for the Month of your choosing!")
    store_num = input("Enter Store ID: ")
    month = input("Enter Month number to predict sales for: ")
    
    # select only StoreID data
    store = walmart.loc[store_num:store_num]
    # sort by weekly sales $ value
    store_sorted = store.sort_values('Weekly_Sales')
    
    # boolean mask for each month from all dates provided from original Walmart data, store_comp is completed table with accurate results
    if month == "2":
        mask = (store_sorted['Date'] >= '2010-2-1') & (store_sorted['Date'] <= '2010-2-28') & (store_sorted['Date'] <= '2011-2-1') & (store_sorted['Date'] <= '2011-2-28')& (store_sorted['Date'] <= '2012-2-1')& (store_sorted['Date'] <= '2012-2-28')
        store_comp = store_sorted.loc[mask]
    if month == "3":
        mask = (store_sorted['Date'] >= '2010-3-1') & (store_sorted['Date'] <= '2010-3-31') & (store_sorted['Date'] <= '2011-3-1') & (store_sorted['Date'] <= '2011-3-31')& (store_sorted['Date'] <= '2012-3-1')& (store_sorted['Date'] <= '2012-3-31')
        store_comp = store_sorted.loc[mask]
    if month == "4":
        mask = (store_sorted['Date'] >= '2010-4-1') & (store_sorted['Date'] <= '2010-4-30') & (store_sorted['Date'] <= '2011-4-1') & (store_sorted['Date'] <= '2011-4-30')& (store_sorted['Date'] <= '2012-2-1')& (store_sorted['Date'] <= '2012-2-28')
        store_comp = store_sorted.loc[mask]
    if month == "5":
        mask = (store_sorted['Date'] >= '2010-5-1') & (store_sorted['Date'] <= '2010-5-31') & (store_sorted['Date'] <= '2011-5-1') & (store_sorted['Date'] <= '2011-5-31')& (store_sorted['Date'] <= '2012-5-1')& (store_sorted['Date'] <= '2012-5-31')
        store_comp = store_sorted.loc[mask]
    if month == "6":
        mask = (store_sorted['Date'] >= '2010-6-1') & (store_sorted['Date'] <= '2010-6-30') & (store_sorted['Date'] <= '2011-6-1') & (store_sorted['Date'] <= '2011-6-30')& (store_sorted['Date'] <= '2012-6-1')& (store_sorted['Date'] <= '2012-6-30')
        store_comp = store_sorted.loc[mask]
    if month == "7":
        mask = (store_sorted['Date'] >= '2010-7-1') & (store_sorted['Date'] <= '2010-7-31') & (store_sorted['Date'] <= '2011-7-1') & (store_sorted['Date'] <= '2011-7-31')& (store_sorted['Date'] <= '2012-7-1')& (store_sorted['Date'] <= '2012-7-31')
        store_comp = store_sorted.loc[mask]
    if month == "8":
        mask = (store_sorted['Date'] >= '2010-8-1') & (store_sorted['Date'] <= '2010-8-31') & (store_sorted['Date'] <= '2011-8-1') & (store_sorted['Date'] <= '2011-8-31')& (store_sorted['Date'] <= '2012-8-1')& (store_sorted['Date'] <= '2012-8-31')
        store_comp = store_sorted.loc[mask]
    if month == "9":
        mask = (store_sorted['Date'] >= '2010-9-1') & (store_sorted['Date'] <= '2010-9-30') & (store_sorted['Date'] <= '2011-9-1') & (store_sorted['Date'] <= '2011-9-30')& (store_sorted['Date'] <= '2012-9-1')& (store_sorted['Date'] <= '2012-9-30')
        store_comp = store_sorted.loc[mask]
    if month == "10":
        mask = (store_sorted['Date'] >= '2010-10-1') & (store_sorted['Date'] <= '2010-10-31') & (store_sorted['Date'] <= '2011-10-1') & (store_sorted['Date'] <= '2011-10-31')& (store_sorted['Date'] <= '2012-10-31')& (store_sorted['Date'] <= '2012-10-31')
        store_comp = store_sorted.loc[mask]
    if month == "11":
        mask = (store_sorted['Date'] >= '2010-11-1') & (store_sorted['Date'] <= '2010-11-30') & (store_sorted['Date'] <= '2011-11-1') & (store_sorted['Date'] <= '2011-11-30')& (store_sorted['Date'] <= '2012-11-1')& (store_sorted['Date'] <= '2012-11-30')
        store_comp = store_sorted.loc[mask]
    if month == "12":
        mask = (store_sorted['Date'] >= '2010-12-1') & (store_sorted['Date'] <= '2010-12-31') & (store_sorted['Date'] <= '2011-12-1') & (store_sorted['Date'] <= '2011-12-31')& (store_sorted['Date'] <= '2012-12-1')& (store_sorted['Date'] <= '2012-12-31')
        store_comp = store_sorted.loc[mask]
    if month == "1":
        mask = (store_sorted['Date'] <= '2010-1-1') & (store_sorted['Date'] <= '2010-1-31')& (store_sorted['Date'] <= '2011-1-1')& (store_sorted['Date'] <= '2011-1-31')& (store_sorted['Date'] <= '2012-1-1')& (store_sorted['Date'] <= '2012-1-31')
        store_comp = store_sorted.loc[mask]
    
    # select predictor columns for Random Forest Procedure
    features_RF = store_comp.columns[np.r_[1,2,11:16]]
    # select predictor columns for Linear Regression
    features_reg = store_comp.columns[np.r_[1,6:9]]
     
    # Random Forest Procedure: Create train and test data
    X_train, X_test, y_train, y_test = train_test_split(store_comp[features_RF], store_comp['sales_class'], test_size=0.4, random_state=0)
    # Linear Regression: train and test data
    X_trainr, X_testr, y_trainr, y_testr = train_test_split(store_comp[features_reg], store_comp['Weekly_Sales'], test_size=.33, random_state=42)
    # Fit train data into classifier for Random Forest
    clf = RandomForestClassifier(n_jobs=2, random_state=0)
    clf.fit(X_train,y_train)
        
    # Predicted sales value ranges
    pred_num = clf.predict(X_test)
    # Departments in the test group, should align with above results
    test_dept = X_test['Dept']
    # table with predicted sales value ranges and corresponding departments
    pred_store = pd.DataFrame({'pred_sales':pred_num, 'Dept':test_dept})
    # Average duplicate departments's sales value
    pred_store_dup = pred_store.groupby('Dept', as_index=False).mean()
    # sort by sales value, descending
    pred_store_sort = pred_store_dup.sort_values('pred_sales', ascending=False)
    # change dept number to corresponding name
    pred_store_sort.replace(dept_change, inplace=True)
    
    # print top and bottom 10 departments
    print("\nThe Predicted Bottom 10 Sales Departments:\n",pred_store_sort['Dept'][-10:])
    print("\nThe Predicted Top 10 Sales Departments are:\n",pred_store_sort['Dept'][:10])
    
    #See how store did in the years of data collected (2010-2012) in average monthly sales.
    
    if store_num == "1":
        df = store.groupby(store['Date'].dt.strftime('%B'))['Weekly_Sales'].mean().sort_values()
        df11 = df.plot(label="store1")
        plt.ylabel('Average Weekly Sales')
        plt.title('Average Weekly Sales for Store 1 2010-2013')
        plt.legend()
        plt.show()
    if store_num == "2":
        df2 = store.groupby(store['Date'].dt.strftime('%B'))['Weekly_Sales'].mean().sort_values()
        df22 = df2.plot(label="store2")
        plt.legend()
        plt.show()
    if store_num == "3":
        df3 = store.groupby(store['Date'].dt.strftime('%B'))['Weekly_Sales'].mean().sort_values()
        df33 = df3.plot(label="store3")
        plt.legend()
        plt.show()
    if store_num == "4":
        df4 = store.groupby(store['Date'].dt.strftime('%B'))['Weekly_Sales'].mean().sort_values()
        df44 = df4.plot(label="store4")
        plt.legend()
        plt.show()
    if store_num == "5":
        df5 = store.groupby(store['Date'].dt.strftime('%B'))['Weekly_Sales'].mean().sort_values()
        df55 = df5.plot(label="store5")
        plt.legend()
        plt.show()
    
    
    #Linear Regression
    clf_reg = linear_model.LinearRegression()
    x_train = poly.fit_transform(X_trainr)
    x_test = poly.fit_transform(X_testr)
    clf_reg.fit(x_train,
           y_trainr)
    #print coefficients of model and results
    #print('Coefficients: \n', clf_reg.coef_)
    y_pred = clf_reg.predict(x_test)
    # Create table for only predicted sales value
    pred_store_reg = pd.DataFrame({'pred_$':y_pred})
    
    #print("Mean squared error: %.2f",
     #    mean_squared_error(y_testr, y_pred))
    #print('R2 score: %.2f' % r2_score(y_testr, y_pred))
    
    # Split predicted Sales values into 4 weeks (according to how many weeks in a month) and find average.
    m = len(y_pred)//4
    week1 = (pred_store_reg['pred_$'][:m]).mean()
    week2 = (pred_store_reg['pred_$'][m:(m+(m+1))]).mean()
    week3 = (pred_store_reg['pred_$'][(m+(m+1)):((m+(m+1))+m)]).mean()
    week4 = (pred_store_reg['pred_$'][((m+(m+1))+m):((m+(m+1))+m)+m]).mean()
    
    week1avg = round(week1,2)
    week2avg = round(week2,2)
    week3avg = round(week3,2)
    week4avg = round(week4,2)
    
    print("\n**Weekly Predicted Sales Summary**")
    print("Week 1 Average Predicted Sales: $",week1avg)
    print("Week 2 Average Predicted Sales: $",week2avg)
    print("Week 3 Average Predicted Sales: $",week3avg)
    print("Week 4 Average Predicted Sales: $",week4avg)
    
    # create bar graph of average weekly predicted sales.
    objects = ('Week 1', 'Week 2', 'Week 3', 'Week 4')
    y_pos = np.arange(len(objects))
    performance = [week1avg,week2avg,week3avg,week4avg]
    fig = plt.figure()
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Average Weekly Sales $')
    plt.title('Average Predicted Weekly Sales')
     
    plt.show()
    fig.savefig(input("Save file as: "))
     
    '''
    #Accuracy for Random Forest
    # Confusion Matrix
    print(confusion_matrix(y_test,pred_num))
    # Cross Validation  Score
    scores = cross_val_score(clf, walmart[features], walmart['sales_class'], cv=5)
    print(scores)
    #view feature/predictor variable importance and their importance scores (higher the better)
    print(list(zip(X_train,clf.feature_importances_)))
    # Avg Cross Validation Score
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    '''
    '''
    # Naive Bayes classifier using Python to predict weekly sales:
    import pandas as pd
    import numpy as np
    # For preprocessing the data
    from sklearn.preprocessing import Imputer
    from sklearn import preprocessing
    # To split the dataset into train and test datasets
    from sklearn.cross_validation import train_test_split
    # To model the Gaussian Navie Bayes classifier
    from sklearn.naive_bayes import GaussianNB
    # To calculate the accuracy score of the model
    from sklearn.metrics import accuracy_score
    
    #Data import: walmart_data.csv is cleaned and transformed into categories by SQL queries
    walmart_df = pd.read_csv('walmart_data.csv', header = None, delimiter=' *, *', engine='python')
    
    # Add headers to the dataframe
    walmart_df.columns = ['Store','IsHoliday','Dept','Type','Temp_class','Fuel_class','Unemploy_class','Cpi_class','Size_class', 'Sales_class']
    
    #test whether there is any null value in our dataset or not
    print(walmart_df.isnull().sum())
    ###The above output shows that there is no “null” value in our dataset.
    
    #Summary statistics of all the attributes.
    print(walmart_df.describe(include= 'all'))
    
    #Use the One-Hot encoder to encode the data into binary format.
    le = preprocessing.LabelEncoder()
    W_Store = le.fit_transform(walmart_df.Store)
    W_IsHoliday = le.fit_transform(walmart_df.IsHoliday)
    W_Dept = le.fit_transform(walmart_df.Dept)
    W_Type = le.fit_transform(walmart_df.Type)
    W_Sales_class = le.fit_transform(walmart_df.Sales_class)
    W_Temp_class = le.fit_transform(walmart_df.Temp_class)
    W_Unemploy_class = le.fit_transform(walmart_df.Unemploy_class)
    W_Cpi_class = le.fit_transform(walmart_df.Cpi_class)
    W_Fuel_class = le.fit_transform(walmart_df.Fuel_class)
    W_Size_class = le.fit_transform(walmart_df.Size_class)
    
    #Initialize the encoded categorical columns
    walmart_df['W_Store'] = W_Store
    walmart_df['W_IsHoliday'] = W_IsHoliday
    walmart_df['W_Dept'] = W_Dept
    walmart_df['W_Type'] = W_Type
    walmart_df['W_Sales_class'] = W_Sales_class
    walmart_df['W_Temp_class'] = W_Temp_class
    walmart_df['W_Unemploy_class'] = W_Unemploy_class
    walmart_df['W_Cpi_class'] = W_Cpi_class
    walmart_df['W_Fuel_class'] = W_Fuel_class
    walmart_df['W_Size_class'] = W_Size_class
    
    #drop the old categorical columns from Walmart data
    dummy_fields = ['Store','IsHoliday','Dept','Type','Temp_class','Fuel_class','Unemploy_class','Cpi_class','Size_class', 'Sales_class']
    walmart_df = walmart_df.drop(dummy_fields, axis = 1)
    
    print(walmart_df.head()) ##The result show that all the columns are not in correct order and should be reindexed. 
    
    # reindexing the columns
    walmart_df = walmart_df.reindex_axis(['W_Store','W_IsHoliday','W_Dept','W_Type','W_Temp_class','W_Fuel_class','W_Unemploy_class','W_Cpi_class','W_Size_class', 'W_Sales_class'], axis= 1)
    
    print(walmart_df.head(1))
    
    #Standardization of Data. We will comvert the data values into standardized values
    num_features = ['W_Store','W_IsHoliday','W_Dept','W_Type','W_Temp_class','W_Fuel_class','W_Unemploy_class', 'W_Cpi_class','W_Size_class']
    
    scaled_features = {}
    for each in num_features:
        mean, std = walmart_df[each].mean(), walmart_df[each].std()
        scaled_features[each] = [mean, std]
        walmart_df.loc[:, each] = (walmart_df[each] - mean)/std
    
    #Split the data into training and test set with sklearn’s train_test_split() 
        
    features = walmart_df.values[:,:9]
    target = walmart_df.values[:,9]
    features_train, features_test, target_train, target_test = train_test_split(features,
                                                                                target, test_size = 0.33, random_state = 42)   
    # The feature set consists of 9 columns i.e, predictor variables and target set consists of 1 column with sales values.
    
    #Gaussian Naive Bayes Implementation. We will be  using sklearn’s GaussianNB module.
    clf = GaussianNB()
    
    #We have built a GaussianNB classifier. The classifier is trained using training data. We can use fit() method for training it
    clf.fit(features_train, target_train)
    
    #After building a classifier, we will use this to make predictions. 
    target_pred = clf.predict(features_test) 
    print(target_pred)
    
    #Accuracy of our Gaussian Naive Bayes model
    #We have made some predictions. Let’s compare the model’s prediction with actual target values for the test set
    acc = accuracy_score(target_test, target_pred, normalize = True)
    print(acc)
    
    """###Our model is giving an accuracy of 65%."""
    
    '''
if __name__ == "__main__":
    main()