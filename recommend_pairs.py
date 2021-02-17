import json
from models.recs import RECS
from utils.utils import plot_recommendations

if __name__ == '__main__':

    with open('parameters.json', 'r') as read_file:

        parameters = json.load(read_file)

        ingredient = parameters['ingredient']
        top = parameters['top']
        plot = parameters['plot']
        save = parameters['save']

    model = RECS(ingredient=ingredient, top=top,
                 flavor_profiles=None, recipes=None)
    recommendations = model.run()

    if plot:
        plot_recommendations(ingredient=ingredient,
                             recommendations=recommendations,
                             top=top,
                             save=save)

    print(recommendations.head(top))
