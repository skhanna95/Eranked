# Eranked
## Improving product search relevance

### Description:
E-commerce is an ever-growing field. In order to provide best customer experience, online retailers have to customize their results in a way such that it suits the customer. In order to give results which are relevant to the customers, the retailers focus on search queries that customer enters. They understand the importance of search relevance and how long and varied results can make their users trust them less, in today’s advancing technology every user wants relevant and quick results.
Motivation:
-	High-quality search is all about returning relevant results even when the data is changing or poorly structured, the queries are imprecise.
-	In order to accustom to changes, we want to make a model which sees a search query and gives relevant products according to the search rankings.
-	Thus, given only raw text as input, our goal is to predict the relevancy of products to search results.
-	In order to achieve this, we will be working on rankings of the products based on search from our dataset.
-	We will be working on to improve the RMSE score based on ranking using different approaches.

### Approach:
Our implementation is divided into 4 phases:
1.	The first phase includes data pre-processing. In order to clean the data we will be remove stop words, use stemming to reduce dictionary size and then finally use scRNN (semi character RNN) to perform spelling correction and removing extra noise.

2.	The next phase is to implement information retrieval rankings:
-	As this problem closely resembles regression (given set of predictor variables, predict the ranking of the product). We will be using Gradient boosting/ XGBoost for regression as our baseline to get appropriate rankings for the search.
-	The next step is to apply CNN -RNN deep learning approach to find relevance of products given search query, for finding relevance. This is done to improve our performance and get better results as deep learning has proven to be a successor over simple regression algorithms.

3.	In order to further improve and get products customized to a given query we will be using regression.
4.	Our final task will be to have a frontend for model serving and sampling.


