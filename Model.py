'''AIR QUALITY PREDICTION using multiple regression.'''

# Importing Libraries
import matplotlib.pyplot as plt
import pandas as pd 
def main():
    # Importing Data
    dataset = pd.read_csv('Train.csv')
    testset = pd.read_csv('Test.csv')
    x_train = dataset.iloc[1:, :-1].values
    y_train = dataset.iloc[1:, -1].values
    x_test = testset.iloc[1:, :].values
    
    # Training the model 
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

    # Predicting the test results
    y_pred = regressor.predict(x_test)
    print(y_pred)
    with open ("Predic.csv",'w') as file:
        for i in y_pred:
            file.write(str(i)+'\n')
    
    

if __name__ == "__main__":
    main()