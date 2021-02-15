import itertools
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.neighbors import NearestNeighbors


class KNNeighbours:

    def __init__(self, flavor_profiles, n):
        self.flavor_profiles = flavor_profiles
        self.n = n

    def fit(self):

        """Initialize and fit KNN to flavor profiles"""

        # build and fit KNN model
        nn = NearestNeighbors(n_neighbors=self.n, metric='jaccard')
        nn.fit(self.flavor_profiles)
        return nn

    def get_neighbors(self, ingredient):

        """Returns neighbors of single ingredient using its flavor profile"""

        nn = self.fit()

        query = self.flavor_profiles.loc[ingredient].to_numpy().reshape(1, -1)
        neighbors_distances, neighbors_indices = nn.kneighbors(query)
        neighbors = self.flavor_profiles.iloc[neighbors_indices[0]].index.to_list()
        similarities = neighbors_distances.tolist()[0]

        neighbors = pd.DataFrame({'Similar food': neighbors,
                                  'Similarity': similarities})

        return neighbors


class PMI:

    def __init__(self, recipes, pmi_scores):
        self.recipes = recipes
        self.pmi_scores = pmi_scores

    def calculate_pmi(self):

        """Calculate occurrences, co-occurrences and PMI for each ingredient pair in recipes"""

        # count ingredients and pairs of ingredients in the recipes
        cooc_counts = Counter()
        ing_count = Counter()
        for ingredients in self.recipes.ingredient_names:
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

        # write counts to dataframe
        pmi_scores = pd.DataFrame(
            {(ingredient_a, ingredient_b,
              ing_count[ingredient_a], ing_count[ingredient_b], ab_count)
             for (ingredient_a, ingredient_b), ab_count in cooc_counts.items()},
            columns=['a', 'b', 'a_count', 'b_count', 'ab_count'])

        # calculate P(A), P(B) and P(A, B) and PMI(A, B) from the ingredient counts
        # P(A) = counts(A) / num of any ingredient occurrence in all recipes
        # P(A, B) = co-occurrences(A, B) / num of any pair of ingredients co-occurrence in all recipes
        p_a = pmi_scores.a_count / sum(ing_count.values())
        p_b = pmi_scores.b_count / sum(ing_count.values())
        p_a_b = pmi_scores.ab_count / sum(cooc_counts.values())
        pmi_scores['PMI'] = np.log(p_a_b / (p_a * p_b))

        # set all negative PMI values to zero
        # set the PMI values for outlier pairs with low co-occurrences to zero
        pmi_scores.loc[pmi_scores['ab_count'] < 5, ['PMI', 'ab_count']] = 0
        pmi_scores.loc[pmi_scores['PMI'] < 0, 'PMI'] = 0

        pmi_scores.sort_values('PMI', ascending=False)

        self.pmi_scores = pmi_scores

        return self.pmi_scores

    def find_combinations(self, ingredient, n):

        """Find n best combinations for ingredient"""

        if self.pmi_scores is None:
            self.calculate_pmi()

        combinations = self.pmi_scores. \
            loc[(self.pmi_scores.a == ingredient) | (self.pmi_scores.b == ingredient)]. \
            sort_values(ascending=False, by='PMI').\
            head(n)

        if combinations.shape[0] == 0:
            return pd.DataFrame({})

        else:
            combinations['Combined food'] = combinations.apply(lambda row: row.a if row.b == ingredient else row.b, axis=1)
            combinations = combinations.loc[:, ['Combined food', 'PMI']]

            return combinations


class RECS:

    def __init__(self, ingredient, encoded_ingredients, recipes, top):
        self.ingredient = ingredient
        self.encoded_ingredients = encoded_ingredients
        self.recipes = recipes
        self.top = top

    def set_flavor_profiles(self):
        self.encoded_ingredients = pd.read_csv('data/processed/encoded_ingredients.csv', index_col=0)

    def set_recipes(self):
        self.recipes = pd.read_csv('data/processed/FN_recipes', index_col=0,
                              converters={"ingredient_names": lambda x: x.strip("[]").replace("'", "").split(", ")})

    def recs_fw(self):

        """firstly looks for pairs of ingredient
             then for neighbors of pairs"""

        # PMI
        pmi_scorer = PMI(recipes=self.recipes,
                         pmi_scores=None)

        # find best combinations for ingredient
        combined_foods = pmi_scorer.find_combinations(ingredient=self.ingredient, n=10)
        combined_foods_set = set(combined_foods['Combined food'])

        # KNN
        model = KNNeighbours(flavor_profiles=self.encoded_ingredients, n=10)

        # find neighbors for combined foods, and their PMIs with ingredient
        neighbors = []
        for combined_food in combined_foods_set:
            neighbors.append(model.get_neighbors(ingredient=combined_food))

        neighbors = pd.concat(neighbors)

        # Combine KNN and PMI
        combined_and_neighbors = pd.merge(left=neighbors,
                                          right=combined_foods,
                                          left_on='Similar food',
                                          right_on='Combined food',
                                          how='left')
        combined_and_neighbors = combined_and_neighbors.fillna(method='ffill')

        # drop rows where ingredient is recommended to itself
        combined_and_neighbors = combined_and_neighbors. \
            loc[combined_and_neighbors['Similar food'] != self.ingredient]. \
            rename(columns={'Similar food': 'Recommended food'})

        recommendations = self.generate_recommendations(combined_and_neighbors)

        return recommendations

    def recs_rev(self):

        """firstly looks for neighbors of ingredient,
                then for pairs of neighbors"""

        # KNN
        model = KNNeighbours(flavor_profiles=self.encoded_ingredients, n=10)

        # find neighbors for ingredient
        neighbors = model.get_neighbors(ingredient=self.ingredient)

        # PMI
        pmi_scorer = PMI(recipes=self.recipes,
                         pmi_scores=None)

        # find best combinations for ingredient neighbors
        combined_foods = [pmi_scorer.find_combinations(ingredient=neighbor, n=10)
                          for neighbor in neighbors['Similar food']]

        # Combine KNN and PMI
        # add similarities
        combined_and_neighbors = pd.concat(combined_foods)
        similarities = [neighbors.Similarity[i]
                        for i in range(len(combined_foods))
                        if combined_foods[i].shape[0] != 0]
        combined_and_neighbors['Similarity'] = [similarities[i // 10] for i in range(combined_and_neighbors.shape[0])]

        # drop rows where ingredient is recommended to itself
        combined_and_neighbors = combined_and_neighbors. \
            loc[combined_and_neighbors['Combined food'] != self.ingredient]. \
            rename(columns={'Combined food': 'Recommended food'})

        recommendations = self.generate_recommendations(combined_and_neighbors)

        return recommendations

    def run(self):

        """Generate recommendations with one or both RECS approaches"""

        self.set_recipes()
        self.set_flavor_profiles()

        recipes_ingredients = set(self.recipes.ingredient_names.explode())
        fn_ingredients = set(self.encoded_ingredients.index)

        if (self.ingredient not in recipes_ingredients) and (self.ingredient not in fn_ingredients):
            print('Ingredient not found in the data set')
            return None

        elif (self.ingredient in recipes_ingredients) and (self.ingredient not in fn_ingredients):
            print('Ingredient found only in recipes, performing forward RECS algorithm')
            recommendations_fw = self.recs_fw()
            return recommendations_fw

        elif (self.ingredient not in recipes_ingredients) and (self.ingredient in fn_ingredients):
            print('Ingredient found only in flavor profiles, performing reverse RECS algorithm')
            recommendations_rev = self.recs_rev()
            return recommendations_rev

        else:
            # when ingredient is both in flavor profiles set and recipes set, combine two RECS approaches
            recommendations_fw = self.recs_fw()
            recommendations_rev = self.recs_rev()

            # combine recommendations
            recommendations = pd.merge(right=recommendations_fw,
                                       left=recommendations_rev,
                                       on='Recommended food',
                                       how='outer').fillna(value=0)
            take_bigger = lambda rating1, rating2: rating1 if (rating1 > rating2) else rating2
            recommendations['Rating'] = recommendations['Rating_x'].combine(recommendations['Rating_y'], take_bigger)

            # remove duplicated recommendations
            recommendations = recommendations. \
                groupby('Recommended food'). \
                Rating.first(). \
                reset_index(). \
                sort_values(by='Rating', ascending=False)
            recommendations = recommendations.loc[:, ['Recommended food', 'Rating']]

            return recommendations

    @staticmethod
    def generate_recommendations(combined_and_neighbors):

        """calculate and normalize rating, return recommendations with ratings"""

        # min-max PMI normalization
        combined_and_neighbors['PMI'] = round(100 * combined_and_neighbors.PMI / combined_and_neighbors.PMI.max(), 2)

        # Rating calculation
        combined_and_neighbors['Rating'] = round(
            (1 - combined_and_neighbors['Similarity']) * combined_and_neighbors['PMI'], 2)

        # remove duplicated recommendations
        combined_and_neighbors = combined_and_neighbors. \
            groupby('Recommended food'). \
            Rating.first(). \
            reset_index(). \
            sort_values(by='Rating', ascending=False)

        recommendations = combined_and_neighbors.copy().loc[:, ['Recommended food', 'Rating']]

        return recommendations


