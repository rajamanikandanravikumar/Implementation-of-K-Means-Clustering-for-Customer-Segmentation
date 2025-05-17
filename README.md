# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import dataset and print head,info of the dataset

2.check for null values

3.Import kmeans and fit it to the dataset

4.Plot the graph using elbow method

5.Print the predicted array

6.Plot the customer segments

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Rajamanikandan R
RegisterNumber:  212223220082
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("Mall_Customers.csv")

print(data.head())
print(data.info())
print(data.isnull().sum())

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(data.iloc[:, 3:5])
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.grid(True)
plt.show()

km = KMeans(n_clusters=5, init="k-means++", random_state=42)
y_pred = km.fit_predict(data.iloc[:, 3:5])
data["Cluster"] = y_pred

plt.figure(figsize=(8, 6))
colors = ['red', 'black', 'blue', 'green', 'magenta']
for i in range(5):
    cluster = data[data["Cluster"] == i]
    plt.scatter(cluster["Annual Income (k$)"], cluster["Spending Score (1-100)"], 
                c=colors[i], label=f"Cluster {i}")

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments")
plt.legend()
plt.grid(True)
plt.show()
```

## Output:
### DATA.HEAD():
![Screenshot 2025-05-17 085422](https://github.com/user-attachments/assets/ad4dd295-9b22-4df0-907a-5d5213e10f8d)

### DATA.INF0():
![Screenshot 2025-05-17 085437](https://github.com/user-attachments/assets/376aa446-2157-4c47-b767-c9f232ec0e46)

### DATA.ISNULL().SUM():
![Screenshot 2025-05-17 085452](https://github.com/user-attachments/assets/e38a25b9-da47-4361-b5bb-86cf0cefd71d)

### PLOT USING ELBOW METHOD:
![Screenshot 2025-05-17 090057](https://github.com/user-attachments/assets/c1404957-ced8-4207-a604-f81a9ef821d1)

### CUSTOMER SEGMENT:
![Screenshot 2025-05-17 090046](https://github.com/user-attachments/assets/60f6d66b-c4cf-4cbb-914c-4e3a8e86d419)



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
