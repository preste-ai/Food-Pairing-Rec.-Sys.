import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from recs_algorithms import calculate_pmi

"""Read data"""
# Molecules content
ingr = pd.read_csv('data/raw/FlavorNetwork/ingr_info.tsv', sep='\t', header=0)
ingr_comp = pd.read_csv('data/raw/FlavorNetwork/ingr_comp.tsv', sep='\t', header=0)

# Recipes
flav_recipes1 = pd.read_csv('data/raw/FlavorNetwork/cuisine-ingredients.csv', skiprows=4, names=list(range(20)), error_bad_lines=False)
flav_recipes2 = pd.read_csv('data/raw/allr_recipes.txt', sep='\t', names=list(range(20)), error_bad_lines=False)

"""Make recipe data set"""
# map countries to regions
map_countries = pd.read_csv('data/raw/map.txt', sep='\t', names=['from', 'to'])
map_countries_dict = map_countries.to_dict()
map_countries_dict = map_countries.set_index('from').to_dict()
map_countries_dict = map_countries_dict['to']
flav_recipes2.iloc[:, 0] = flav_recipes2.iloc[:, 0].map(map_countries_dict)

# concatenate two flav_network datasets
flav_recipes = pd.concat([flav_recipes1, flav_recipes2]).drop_duplicates()

# create recipes df with the list of ingredients for each recipe
recipes = pd.DataFrame({})
recipes['country'] = flav_recipes.iloc[:, 0]
recipes['ingredient_names'] = flav_recipes.iloc[:, 1:].apply(lambda row: row.dropna().tolist(), axis=1)

"""Filter data sets"""
# drop exotic countries
exotic_cuisines = ['African', 'SoutheastAsian', 'SouthAsian', 'EastAsian']
recipes = recipes[~ recipes.country.isin(exotic_cuisines)]

# filter for alcohol
alcohol = ingr.loc[ingr.category == 'alcoholic beverage', 'ingredient name']
filter_alcohol = lambda recipe: not any(beverage in recipe for beverage in alcohol)
recipes = recipes[recipes['ingredient_names'].apply(filter_alcohol)]

# drop useless ingredients from the data set
useless_ingredients = ['nut', 'leaf', 'tea', 'fruit', 'condiment', 'wheat', 'vegetable',
                       'dairy', 'berry', 'seed', 'date', 'plant', 'clam', 'flower', 'wood', 'root']
filter_ingredients = lambda ingredient: True if ingredient not in useless_ingredients else False
recipes['ingredient_names'] = recipes.ingredient_names.apply(lambda x: list(filter(filter_ingredients, x)))

# drop recipes with less than 3 ingredients
filter_len = lambda lst: lst if len(lst) > 2 else np.nan

recipes['ingredient_names'] = recipes['ingredient_names'].apply(filter_len)
recipes.dropna(subset=['ingredient_names'], inplace=True)

"""Calculate PMI"""
pmi_df = calculate_pmi(recipes.ingredient_names)

"""Make flavor profiles"""
ingr_ids = ingr.loc[:, ['# id', 'ingredient name']]
ingr_comp_lists = ingr_comp.groupby('# ingredient id')['compound id'].apply(list).reset_index()
n_compounds = ingr_comp['compound id'].nunique()

flavors_df = pd.merge(left=ingr_ids, right=ingr_comp_lists, left_on='# id', right_on='# ingredient id').\
    drop(['# id', '# ingredient id'], axis=1).\
    rename(columns={'compound id': 'compounds'}).\
    set_index('ingredient name').\
    sort_index()

# encode flavor profiles to get the flavor vectors with the same length
binarizer = MultiLabelBinarizer(classes=range(n_compounds))
encoded_ingredients = pd.DataFrame(binarizer.fit_transform(flavors_df.compounds), index=flavors_df.index)
encoded_ingredients = encoded_ingredients == 1


recipes.to_csv('data/processed/FN_recipes.csv')
pmi_df.to_csv('data/processed/pmi_flavornetwork.csv')
encoded_ingredients.to_csv('data/processed/encoded_ingredients.csv')