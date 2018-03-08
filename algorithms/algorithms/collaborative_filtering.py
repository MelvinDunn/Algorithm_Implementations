import numpy as np
import pandas as pd
import itertools

def item_to_item_similarity(user_item):
	#cosine similarity between items that have been rated between users.
	
	#get the different combinations of possible ratings.
	combos = (list(itertools.combinations(list(user_item.columns), 2)))

	#make an empty dataframe for them.
	item_to_item_df = pd.DataFrame({}, index = user_item.columns, columns = user_item.columns)
	for i in combos:
		index_1 = (~pd.isnull(user_item[i[0]]))
		index_2 = (~pd.isnull(user_item[i[1]]))
		master_index = (index_1 == index_2)
		item_to_item_df.loc[i[0],i[1]] = (cosine(user_item[i[0]][master_index],user_item[i[1]][master_index]))
	item_to_item_df = item_to_item_df.T.fillna(0) + item_to_item_df.fillna(0)	
	return item_to_item_df + np.diag([1 for i in range(item_to_item_df.shape[0])])

def predict_missing_ratings(user_item):
	item_to_item_df = item_to_item_similarity(user_item)
	for i in range(user_item.shape[0]):
		rated_items = (user_item.iloc[i].dropna())
		rated_items_index = rated_items.index
		if len(rated_items_index) < user_item.shape[1]:
			potential_columns = (item_to_item_df.loc[rated_items_index] != 1)
			columns_sum = np.sum(potential_columns, axis=0)
			item_vec = np.asarray(item_to_item_df[(columns_sum == max(columns_sum))][rated_items_index])
			rated_items = np.asarray(rated_items)
			predictions = np.dot(item_vec, rated_items) / np.sum(item_vec)
			user_item.iloc[i] = user_item.iloc[i].fillna(predictions[0])
	return user_item

def cosine(a,b): 
	return np.dot(a,b) / (np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2)))

if __name__ == "__main__":

	data = pd.DataFrame({"ID":[241,222,276,273,200,229,231,239,286],
			"user": ["u1", "u1", "u2","u2","u3","u3","u3","u4","u4"],
			"item": ["m1","m3","m1","m2","m1","m2","m3","m2","m3"],
			"rating": [2,3,5,2,3,3,1,2,2]})
	#create a user-item crosstable.
	user_item = (pd.crosstab(index=data["user"],columns=data["item"],values=data["rating"], aggfunc="mean"))
	#return the ratings, the way I'm filling in the missing values isn't the proper way.
	#As a change I would have to flatten the matrix and iterate through the NAs.
	print(predict_missing_ratings(user_item))