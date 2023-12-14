# Product-Duplicate-Detection
In this assignment the scalability issue for product duplicate detection is addressed, where we use a data set of four different webshops offering different televisions. The aim is to improve the duplicate detection, while minimizing the number of comparisons. 

## data_cleaning577305
In data_cleaning577305, a dataframe is created, with modelID, title, shop, brand and resolution. Then, if a product has a value None for one of the key-value pairs (shop, brand or resolution), then the title is used to search for this. 
The title is cleaned by replacing all inch and hertz signs by "inch" and "hz". The website is removed from the title, and everything is transformed to lower cases. 
The duplicates counter counts the amount of duplicates, when looking at the modelID. The counter then returns the amount of real duplicates and a matrxix where a value is 1 if it is a duplicate and 0 elsewhere. This can be used later on to check how many of our duplicates are real duplicates. 

## signature_matrix577305
To obtain the binary vector, for the title and brand the model words are extracted and stored in a list of modelwords. We then create a shingle of every model word and add it to a list of unique model words, if it did not occur yet in this list of unique model words. 
To prepare for using LSH, minhashing is first called, where only a fraction of the rows is used. It returns the signature matrix

## LSH577305
Here, LSH is applied. It gets as input a certain threshold and the signature matrix, thus based on this threshold the best number of rows and bands can be caluclated. This holds because the length of the columns of the signature matrix is equal to the amount of rows times the amount of bands. The employed hash function maps the vector to a specific bucket, and this mapping is determined by the sequence (string) of elements (numbers) within the band. This hashing is performed for all products and bands. Products that are hashed to the same bucket are regarded as candidate pairs, which will be further examined during MSM.

## MSM577305
It starts with the construction of a dissimilarity matrix. In this matrix, pairs of products labeled as no possible pair during LSH, are assigned a high value, namely 1000. Besides that, if the brands of the two products are different, or the webshop is the same, or the products have different resolutions, then it is also assigned a high value. The remaining dissimilarity values are computed by using a simlarity function. This function is only used when the specific pair of products was assigned as possible duplicate during LSH. Then, it calculates the Jaccard dissimilarity (1 - Jaccard similarity) between sets of shingles derived from the titles of the two products. The value of this pair in the dissimilarity matrix thus gets the Jaccard dissimilarity value. 
After this, AgglomerativeClustering is used, with the settings: no predfined number of clusters, the distance metric is the dissimilarity matrix and it identifies clusters based on the maximum pairwise dissimilarity between them. 

## evaluation577305
The performance measures for LSH has a different method than for MSM, since LSH still contains a matrix with duplicates. If the sum of a value of the real duplicate matrix and the candidate duplicates is equal to 2, then it means that we correctly classified this duplicate. The pair quality is the nmber of correct classifications divided by the number of comparisons. F1 star is the harmonic mean between pair quality and pair completeness. 
For MSM, the amount of duplicates is counted, to calculate the precision and recall. The F1 measure is the harmonic mean between precision and recall.

## main577305
This is the main, where all methods are called and used to determine the clustering duplicates, and evaluate the perfomances. This is done on the complete data set, and afterwards plots are made. 

## bootstrapCode577305
This is for when bootstrapping is used. It creates the training and test dataframe, where we use sampling with replacement. This results in around 63% of the original data. 

## bootstraps577305
This does the same as in main, it runs the code, but then for 5 bootstraps and for every treshold value t between 0 and 1 (in steps of 0.05). It aggregates the performance measures over the bootstraps and makes a plot out of it. 




