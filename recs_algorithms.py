import pandas as pd
import numpy as np
import pandas as pd
import itertools
from collections import Counter
from sklearn.neighbors import NearestNeighbors


def calculate_pmi(ingredients_series):

    """calculate occurrences, co-occurrences and PMI for each ingredient pair and return them in df"""

    # count ingredients and pairs of ingredients in the recipes
    cooc_counts = Counter()
    ing_count = Counter()
    for ingredients in ingredients_series:
        for ing in ingredients:
            ing_count[ing] += 1
        for (ing_a, ing_b) in itertools.combinations(set(ingredients), 2):
            if ing_a > ing_b:
                ing_a, ing_b = ing_b, ing_a
            cooc_counts[(ing_a, ing_b)] += 1

    # add pairs which are not occurring together
    ingredients = ing_count.keys()
    for (ing_a, ing_b) in itertools.combinations(set(ingredients), 2):
        if ((ing_a, ing_b) not in cooc_counts.keys()) and ((ing_b, ing_a) not in cooc_counts.keys()):
            cooc_counts[(ing_a, ing_b)] = 0

    # transform to DF
    pmi_df = pd.DataFrame(
            {(ingredient_a, ingredient_b,
                ing_count[ingredient_a], ing_count[ingredient_b], ab_count)
                for (ingredient_a, ingredient_b), ab_count in cooc_counts.items()},
            columns=['a', 'b', 'a_count', 'b_count', 'ab_count'])

    # We calculate P(A), P(B) and P(A, B) and PMI(A, B) from the previous df.
    # P(A) is counts(A) / num of any ingredient occurrence in all recipes
    # P(A, B) is coocs(A, B) / num of any pair of ingredients co-occurrence in all recipes
    p_a = pmi_df.a_count / sum(ing_count.values())
    p_b = pmi_df.b_count / sum(ing_count.values())
    p_a_b = pmi_df.ab_count / sum(cooc_counts.values())
    pmi_df['pmi'] = np.log(p_a_b / (p_a * p_b))

    # all negative PMIs & all outliers PMIs are set to zero
    # instead of dropping away the outliers with low co-occurrences, we set the PMI value for these pairs to zero
    pmi_df.loc[pmi_df['ab_count'] < 5, ['pmi', 'ab_count']] = 0
    pmi_df.loc[pmi_df['pmi'] < 0, 'pmi'] = 0

    pmi_df = pmi_df.sort_values('pmi', ascending=False)

    return pmi_df


def get_neighbors(ingredient, ingredients_list, encoded_ingredients, pmi_df, nearest):

    """Returns neighbors of best combined ingredients using their flavor profiles"""

    # build KNN to find ingredient similar to the ones in combinations
    nn = NearestNeighbors(n_neighbors=nearest, metric='jaccard')
    nn.fit(encoded_ingredients)

    # find NN for combined foods
    neighbors = []
    similarities = []
    ratings = []
    for food in ingredients_list:
        query = encoded_ingredients.loc[food].to_numpy().reshape(1, -1)
        neighbors_distances, neighbors_indices = nn.kneighbors(query)
        neighbors.append(encoded_ingredients.
                         iloc[neighbors_indices[0]].
                         index.to_list())
        similarities.append(neighbors_distances.tolist()[0])
        ratings.append(pmi_df.loc[((pmi_df.a == ingredient) & (pmi_df.b == food)) |
                                  ((pmi_df.b == ingredient) & (pmi_df.a == food)), 'pmi'].values)

    ratings = [rating[0] if len(rating) > 0 else 0 for rating in ratings]

    neighbors = pd.DataFrame({'Well combined': ingredients_list,
                              'Rating': ratings,
                              'Similar food': neighbors,
                              'Similarity': similarities})


    return neighbors


def get_neighbors_rev(ingredient, encoded_ingredients, nearest):

    """Returns neighbors of query ingredient using their flavor profiles"""

    # build KNN to find ingredient similar to the ones in combinations
    nn = NearestNeighbors(n_neighbors=nearest, metric='jaccard')
    nn.fit(encoded_ingredients)

    # find NN for combined foods

    query = encoded_ingredients.loc[ingredient].to_numpy().reshape(1, -1)
    neighbors_distances, neighbors_indices = nn.kneighbors(query)
    neighbors = encoded_ingredients.iloc[neighbors_indices[0]].index.to_list()
    similarities = neighbors_distances.tolist()[0]

    neighbors = pd.DataFrame({'Similar food': neighbors,
                              'Similarity': similarities})

    return neighbors


def recsys_fw(ingredient, encoded_ingredients, pmi_df):

    """Returns recommendations dataframe for query ingredient
    firstly looks for pairs
    then for neighbors of pairs"""

    """PMI"""

    # find best combinations
    combinations = pmi_df.loc[(pmi_df.a == ingredient) | (pmi_df.b == ingredient)].sort_values(ascending=False,
                                                                                               by='pmi')
    combinations['Well combined'] = combinations.apply(lambda row: row.b if row.a == ingredient else row.a, axis=1)
    combinations = combinations.head(10)
    combinations_list = list(set(combinations['Well combined']).difference(ingredient))

    """KNN"""

    # find neighbors for combined foods
    neighbors = get_neighbors(ingredient=ingredient,
                              ingredients_list=combinations_list,
                              encoded_ingredients=encoded_ingredients,
                              pmi_df=pmi_df,
                              nearest=5)

    # generate recommendations DF with ratings
    # stack list of neighbor foods and similarities
    paired_food = neighbors.apply(lambda x: pd.Series(x['Similar food']), axis=1).stack().reset_index(level=1,
                                                                                                      drop=True)
    paired_food.name = 'Recommended food'
    similarity = neighbors.apply(lambda x: pd.Series(x['Similarity']), axis=1).stack().reset_index(level=1, drop=True)
    similarity.name = 'Similarity'

    similar_foods = pd.concat([paired_food, similarity], axis=1)

    recommendations = neighbors.drop(['Similar food', 'Similarity'], axis=1).join(similar_foods).sort_values(
        by='Rating', ascending=False)

    # rating normalization
    recommendations['Rating'] = round(100 * recommendations.Rating / recommendations.Rating.max(), 2)

    recommendations['Resulting rating'] = round((1 - recommendations['Similarity']) * recommendations['Rating'], 2)
    recommendations = recommendations. \
        loc[recommendations['Recommended food'] != ingredient]. \
        drop(['Rating', 'Similarity'], axis=1). \
        sort_values(by='Resulting rating', ascending=False)

    # if food is recommended several times for different pairs, take its max rating
    recommendations = recommendations.groupby(['Recommended food']) \
        [['Well combined', 'Resulting rating']].max(). \
        sort_values(by='Resulting rating', ascending=False). \
        reset_index()

    recommendations = recommendations.loc[:, ['Well combined', 'Resulting rating']]. \
        rename(columns={'Well combined': 'Recommended food', 'Resulting rating': 'Rating'})

    return recommendations


def recsys_rev(ingredient, encoded_ingredients, pmi_df):

    """Returns recommendations dataframe for query ingredient
    firstly looks for neighbors
    then for pairs of neigbours"""

    """KNN"""

    # find neighbors for combined foods
    neighbors = get_neighbors_rev(ingredient=ingredient,
                                  encoded_ingredients=encoded_ingredients,
                                  nearest=10)

    """PMI"""

    # find best combinations
    top = 10
    combinations = [pmi_df.
                        loc[(pmi_df.a == ingredient) | (pmi_df.b == ingredient)].
                        sort_values(ascending=False, by='pmi').
                        head(top)
                    for ingredient in neighbors['Similar food']]

    combinations = pd.concat(combinations)

    combinations['Neighbor'] = combinations.\
        apply(lambda row: row.a if row.a in neighbors['Similar food'].to_list() else row.b, axis=1)
    combinations['Well combined'] = combinations.\
        apply(lambda row: row.b if row.a in neighbors['Similar food'].to_list() else row.a, axis=1)

    """Final rating"""

    # combine similarities with pmi, normalize

    recommendations = combinations.copy().loc[:, ['Neighbor', 'Well combined', 'pmi']].rename(columns={'pmi':'Rating'})
    recommendations = pd.merge(left=recommendations,
                               right=neighbors.loc[:, ['Similarity', 'Similar food']],
                               left_on='Neighbor',
                               right_on='Similar food').drop('Similar food', axis=1)
    recommendations['Resulting rating'] = 100 * recommendations['Rating'] / recommendations['Rating'].max() * (1 - recommendations['Similarity'])

    # if food is recommended several times for different neighbors, take its max rating
    recommendations = recommendations.groupby(['Well combined'])[['Neighbor', 'Resulting rating']].max().\
        sort_values(by='Resulting rating', ascending=False). \
        reset_index()

    recommendations = recommendations.loc[:, ['Well combined', 'Resulting rating']]. \
        rename(columns={'Well combined': 'Recommended food', 'Resulting rating': 'Rating'})

    return recommendations


