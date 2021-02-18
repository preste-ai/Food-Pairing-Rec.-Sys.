import pandas as pd
from sklearn.neighbors import NearestNeighbors


class KNNeighbours:

    def __init__(self, flavor_profiles, n):

        """
        :param flavor_profiles: dataframe with flavor profiles of foods represented as boolean vectors of equal length
        :param n: number of neighbors to return
        :param model: KNN model fitted to flavor profiles, returned by get_model method
        """

        self.flavor_profiles = flavor_profiles
        self.n = n
        self.model = None

    def get_model(self):

        """
        Initialize and fit KNN to flavor profiles
        :return: model fitted to flavor profiles
        """

        # build and fit KNN model
        self.model = NearestNeighbors(n_neighbors=self.n, metric='jaccard')
        self.model.fit(self.flavor_profiles)

        return self.model

    def get_neighbors(self, ingredient):

        """
        Find neighbors of ingredient using its flavor profile
        :param ingredient: name of ingredient for which neighbors to be found
        :return: dataframe with neighbors of ingredient
        """

        self.get_model()

        query = self.flavor_profiles.loc[ingredient].to_numpy().reshape(1, -1)
        neighbors_distances, neighbors_indices = self.model.kneighbors(query)
        neighbors = self.flavor_profiles.iloc[neighbors_indices[0]].index.to_list()
        similarities = neighbors_distances.tolist()[0]

        neighbors = pd.DataFrame({'Similar food': neighbors,
                                  'Similarity': similarities})

        return neighbors
