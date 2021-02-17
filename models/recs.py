import pandas as pd
from models.pmi import PMI
from models.knn import KNNeighbours


class RECS:

    def __init__(self, ingredient, flavor_profiles, recipes, top):

        """
        :param ingredient: string name of ingredient for which recommendations should be generated
        :param flavor_profiles: dataframe with flavor profiles of foods represented as boolean vectors of equal length
        :param recipes: data set with recipes represented as lists of ingredient names
        :param top: number of foods in output recommendation list
        """

        self.ingredient = ingredient
        self.encoded_ingredients = flavor_profiles
        self.recipes = recipes
        self.top = top

    def set_flavor_profiles(self):
        self.encoded_ingredients = pd.read_csv('data/encoded_ingredients.csv', index_col=0)

    def set_recipes(self):
        self.recipes = pd.read_csv('data/FN_recipes',
                                   index_col=0,
                                   converters={"ingredient_names": lambda x: x.strip("[]").replace("'", "").split(", ")})

    def recs_fw(self):

        """
        Forward RECS algorithm: firstly looks for pairs of ingredient then for neighbors of pairs
        :return: recommendations dataframe with recommended food and their ratings
        """

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

        """
        Reverse RECS algorithm: firstly looks for neighbors of ingredient, then for pairs of neighbors
        :return: recommendations dataframe with recommended food and their ratings
        """

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

        """
        Generate recommendations with one or both RECS algorithms
        :return: recommendations dataframe with recommended food and their ratings
        """

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

        """
        Calculate and normalize rating, return recommendations with ratings
        :param combined_and_neighbors: dataframe with recommended food
               having PMI and Similarity values that should be combined
        :return: recommendations dataframe with recommended food and calculated ratings
        """

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


