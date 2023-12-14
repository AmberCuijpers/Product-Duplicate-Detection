# Product-Duplicate-Detection
In this assignment the scalability issue for product duplicate detection is addressed, where we use a data set of four different webshops offering different televisions. The aim is to improve the duplicate detection, while minimizing the number of comparisons. 

## data_cleaning577305
In data_cleaning577305, a dataframe is created, with modelID, title, shop, brand and resolution. Then, if a product has a value None for one of the key-value pairs (shop, brand or resolution), then the title is used to search for this. 
The title is cleaned by replacing all inch and hertz signs by "inch" and "hz". The website is removed from the title, and everything is transformed to lower cases. 
The duplicates counter counts the amount of duplicates, when looking at the modelID. The counter then returns the amount of real duplicates and a matrxix where a value is 1 if it is a duplicate and 0 elsewhere. This can be used later on to check how many of our duplicates are real duplicates. 



