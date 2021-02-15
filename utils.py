import seaborn as sns
import matplotlib.pyplot as plt


def plot_recommendations(ingredient, recommendations, top):

    """plot top n recommendations for ingredient"""

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
    plt.show()