import sys
from models import RECS
from utils import plot_recommendations

if __name__ == '__main__':

    ingredient = sys.argv[1]
    top = int(sys.argv[2])

    model = RECS(ingredient=ingredient, top=top,
                 encoded_ingredients=None, recipes=None)
    recommendations = model.run()

    plot_recommendations(ingredient=ingredient,
                         recommendations=recommendations,
                         top=top)

    print(recommendations.head(top))
