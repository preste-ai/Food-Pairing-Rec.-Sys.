import itertools
import numpy as np
import pandas as pd
from collections import Counter


class PMI:

    def __init__(self, recipes, pmi_scores):

        """
        :param recipes: data set with recipes represented as lists of ingredient names
        :param pmi_scores: dataframe with PMI values calculated with calculate_pmi method call
        """

        self.recipes = recipes
        self.pmi_scores = pmi_scores

    def calculate_occurrences(self):

        """
        Calculate occurrences and co-occurrences for each ingredient pair in recipes
        :return: dataframe with occurrence (a_count, b_count) of each ingredient (a, b) in recipes
                 and their co-occurrences ab_count
        """

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
        occurrences = pd.DataFrame(
            {(ingredient_a,
              ingredient_b,
              ing_count[ingredient_a],
              ing_count[ingredient_b],
              ab_count)
             for (ingredient_a, ingredient_b), ab_count
             in cooc_counts.items()},
            columns=['a', 'b', 'a_count', 'b_count', 'ab_count'])

        num_ingredients, num_ingredient_pairs = sum(ing_count.values()), sum(cooc_counts.values())

        return occurrences, num_ingredients, num_ingredient_pairs

    def calculate_pmi(self):

        """
        Calculate probabilities of occurrences and co-occurrences
        and derived Point-wise Mutual Information (PMI) values for each ingredient pair in recipes
        :return: dataframe of ingredients with calculated PMI values
        """

        occurrences, num_ingredients, num_ingredient_pairs = self.calculate_occurrences()

        pmi_scores = occurrences.copy()
        # calculate P(A), P(B) and P(A, B) and PMI(A, B) from the ingredient counts
        # P(A) = counts(A) / num of any ingredient occurrence in all recipes
        # P(A, B) = co-occurrences(A, B) / num of any pair of ingredients co-occurrence in all recipes

        p_a = pmi_scores.a_count / num_ingredients
        p_b = pmi_scores.b_count / num_ingredients
        p_a_b = pmi_scores.ab_count / num_ingredient_pairs
        pmi_scores['PMI'] = np.log(p_a_b / (p_a * p_b))

        # set all negative PMI values to zero
        # set the PMI values for outlier pairs with low co-occurrences to zero
        pmi_scores.loc[pmi_scores['ab_count'] < 5, ['PMI', 'ab_count']] = 0
        pmi_scores.loc[pmi_scores['PMI'] < 0, 'PMI'] = 0

        pmi_scores.sort_values('PMI', ascending=False)
        pmi_scores = pmi_scores.loc[:, ['a', 'b', 'PMI']]

        self.pmi_scores = pmi_scores

        return self.pmi_scores

    def find_combinations(self, ingredient, n):

        """
        Find n best combinations for ingredient
        :param ingredient: string name of ingredient for which pairs should be found
        :param n: number of pairs to be found
        :return: dataframe with n best pairs for ingredient and PMI values between ingredient and each of them
        """

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

