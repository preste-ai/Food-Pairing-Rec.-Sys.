import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from recs_algorithms import recsys_fw, recsys_rev


def plot_recommendations(recommendations):

    """PLOT recommendations"""

    # plot recommendations combined
    nice_palette = sns.cubehelix_palette(n_colors=top,
                                         start=1.3, rot=-1, gamma=1.0, hue=1, light=0.9, dark=0.3, reverse=True)

    plt.figure(figsize=(10, 10))
    sns.barplot(data=recommendations.head(top), dodge=False, x='Recommended food', y='Rating',
                palette=nice_palette)
    plt.xticks(rotation=90)
    plt.xlabel("")
    plt.title('Food pairs for {} using combined RECS'.format(ingredient))
    plt.tight_layout()
    plt.show()


def run_recs(ingredient, top):

    """Generate recommendations with one or both RECS approaches"""

    pmi_df = pd.read_csv('data/processed/pmi_flavornetwork.csv', index_col=0)
    encoded_ingredients = pd.read_csv('data/processed/encoded_ingredients.csv', index_col=0)

    recipes_ingredients = set(pmi_df.a).union(pmi_df.b)
    fn_ingredients = set(encoded_ingredients.index)

    if (ingredient not in recipes_ingredients) and (ingredient not in fn_ingredients):
        print('Ingredient not found in the data set')
        return None

    elif (ingredient in recipes_ingredients) and (ingredient not in fn_ingredients):

        print('Ingredient found only in recipes, performing forward RECS algorithm')

        recommendations_fw = recsys_fw(ingredient=ingredient,
                                       pmi_df=pmi_df,
                                       encoded_ingredients=encoded_ingredients)

        plot_recommendations(recommendations_fw)

        return recommendations_fw

    elif (ingredient not in recipes_ingredients) and (ingredient in fn_ingredients):

        print('Ingredient found only in flavor profiles, performing reverse RECS algorithm')

        recommendations_rev = recsys_rev(ingredient=ingredient,
                                         pmi_df=pmi_df,
                                         encoded_ingredients=encoded_ingredients)

        plot_recommendations(recommendations_rev)

        return recommendations_rev

    else:

        recommendations_fw = recsys_fw(ingredient=ingredient,
                                       pmi_df=pmi_df,
                                       encoded_ingredients=encoded_ingredients)

        recommendations_rev = recsys_rev(ingredient=ingredient,
                                         pmi_df=pmi_df,
                                         encoded_ingredients=encoded_ingredients)

        # combine recommendations
        rec_food = list(set(pd.concat([recommendations_fw['Recommended food'], recommendations_rev['Recommended food']])))

        ratings_rev = [recommendations_rev.loc[recommendations_rev['Recommended food'] == food, 'Rating'].values for food in rec_food]
        ratings_rev = list(map(lambda x: x[0] if len(x) > 0 else 0, ratings_rev))

        ratings_fw = [recommendations_fw.loc[recommendations_fw['Recommended food'] == food, 'Rating'].values for food in rec_food]
        ratings_fw = list(map(lambda x: x[0] if len(x) > 0 else 0, ratings_fw))

        combined_ratings = [max(x, y) for x, y in zip(ratings_fw, ratings_rev)]

        combined_recommendations = pd.DataFrame({'Recommended food':rec_food, 'Rating':combined_ratings})
        combined_recommendations = combined_recommendations.sort_values(by='Rating', ascending=False)

        plot_recommendations(combined_recommendations)

        return combined_recommendations


if __name__ == '__main__':

    import sys

    ingredient = sys.argv[1]
    top = int(sys.argv[2])
    recommendations = run_recs(ingredient, top)
    print(recommendations.head(top))
