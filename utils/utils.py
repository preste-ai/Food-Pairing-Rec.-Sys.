import seaborn as sns
import matplotlib.pyplot as plt


def plot_recommendations(ingredient, recommendations, top, save):

    """
    Plot top n recommendations for ingredient
    :param ingredient: name of ingredient for which recommendations are generated
    :param recommendations: recommendations dataframe with recommended food and their ratings
    :param top: number of recommendations to show
    :param save: boolean to indicate whether figure should be saved
    :return: figure with ratings of recommended food, file with figure if save=True
    """

    # plot recommendations combined
    nice_palette = sns.cubehelix_palette(n_colors=top,
                                         start=1.3,
                                         rot=-1,
                                         gamma=1.0,
                                         hue=1,
                                         light=0.9,
                                         dark=0.3,
                                         reverse=True)

    plt.figure(figsize=(10, 10))
    sns.barplot(data=recommendations.head(top),
                dodge=False,
                x='Recommended food',
                y='Rating',
                palette=nice_palette)
    plt.xticks(rotation=90)
    plt.xlabel("")
    plt.title('Food pairs for {}'.format(ingredient))
    plt.tight_layout()

    if save:
        plt.savefig('recommendations.png')

    plt.show()

