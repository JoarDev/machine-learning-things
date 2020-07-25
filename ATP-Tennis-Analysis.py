#import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Reads the csv file and creates a data frame
df = pd.read_csv('./rsc/tennis_stats.csv')

#Independent variables
x = df[['FirstServe','FirstServePointsWon','SecondServePointsWon','BreakPointsFaced','BreakPointsSaved','ServiceGamesPlayed','ServiceGamesWon','TotalServicePointsWon','FirstServeReturnPointsWon','SecondServeReturnPointsWon','BreakPointsOpportunities','BreakPointsConverted','ReturnGamesPlayed','ReturnGamesWon','ReturnPointsWon','TotalPointsWon']]
#x=df[['ServiceGamesPlayed','ServiceGamesWon']]

#Dependent variable (Outcome)
y=df[['Winnings']]

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8)
idk = LinearRegression()
idk.fit(x_train,y_train)
print(idk.score(x_test,y_test)) #R**2 value
y_predict=idk.predict(x_test)

#Actual values against predicted ones on a scatterplot
plt.scatter(y_test,y_predict,alpha=0.4)

plt.show()