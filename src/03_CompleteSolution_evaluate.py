import nltk
import pandas as pd

###############################################################################
#####                              Corpus                                 #####
###############################################################################
df = pd.read_csv("data/processed/sentiment_with_lemmas.tsv", sep="\t")

ratings = list(df["rating"])
reviews = list(df["review"])
reviews = [str(e) for e in reviews]
lemmatized = list(df["lemmas"])
lemmatized = [str(e).split() for e in lemmatized]
lemmatized = [[e[0] + "_" + e[1] for e in list(nltk.bigrams(e))] for e in lemmatized]
sentiment = list(df["sentiment"])

###############################################################################
#####               Sentiment values instead of scores                    #####
###############################################################################


def get_rating_class(rating):
    if rating > 3:
        return 2
    elif rating == 3:
        return 1
    else:
        return 0


def get_sentiment_value(sentiment):
    if sentiment > 0.2:
        return 2
    elif -0.2 <= sentiment <= 0.2:
        return 1
    else:
        return 0


def check_status(e):
    if e[0] == e[1]:
        return "OK"
    else:
        return "CHECK"


rating_classes = [get_rating_class(e) for e in ratings]
sentiment_values = [get_sentiment_value(e) for e in sentiment]

# just for marking those reviews that should be checked
check = [check_status(e) for e in zip(rating_classes, sentiment_values)]


new_df = pd.DataFrame(
    {
        "ratings": ratings,
        "sentiment_value": sentiment_values,
        "rating_class": rating_classes,
        "status": check,
        "reviews": reviews,
    }
)

with open("data/processed/annotation.tsv", "w") as outfile:
    outfile.write(new_df.to_csv(index=False, sep="\t"))

# the proportion of correctly categorized reviews
print(len([e for e in check if e == "OK"]) / len(check))

training_df = pd.DataFrame({"reviews": reviews, "rating_class": rating_classes})


with open("data/processed/training.tsv", "w") as outfile:
    outfile.write(training_df.to_csv(index=False, sep="\t"))
###############################################################################
#####                       Evaluation                                    #####
###############################################################################
# evaluation
from sklearn.metrics import accuracy_score

acc = accuracy_score(rating_classes, sentiment_values)
print(acc)

from sklearn.metrics import classification_report

target_names = ["negative", "neutral", "positive"]
print(
    classification_report(rating_classes, sentiment_values, target_names=target_names)
)

#              precision    recall  f1-score   support
#     negative       0.86      0.05      0.09      2000
#      neutral       0.21      0.94      0.35      1000
#     positive       0.87      0.22      0.35      2000
#     accuracy                           0.29      5000
#    macro avg       0.65      0.40      0.26      5000
# weighted avg       0.73      0.29      0.24      5000

import altair as alt
import numpy as np
from sklearn.metrics import confusion_matrix

x, y = np.meshgrid(range(0, 3), range(0, 3))
cm = confusion_matrix(rating_classes, sentiment_values, labels=[0, 1, 2])

source = pd.DataFrame({"true": x.ravel(), "predicted": y.ravel(), "number": cm.ravel()})

chart = (
    alt.Chart(source)
    .mark_rect()
    .encode(x="true:O", y="predicted:O", color="number:Q", tooltip=["number"])
    .interactive()
    .properties(width=800, height=500)
)
chart.save("plots/05/confusion_matrix.html")
