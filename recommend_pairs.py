import sys
from utils import run_recs

if __name__ == '__main__':

    ingredient = sys.argv[1]
    top = int(sys.argv[2])
    recommendations = run_recs(ingredient, top)
    print(recommendations.head(top))
