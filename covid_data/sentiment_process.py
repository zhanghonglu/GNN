import pandas as pd

from covid_data.usa_state import name_to_state

sentiment = pd.read_csv('../res/post.csv', usecols=['state','sentiment'])
sentiment_num = pd.concat([sentiment["state"],pd.to_numeric(sentiment["sentiment"], errors='coerce')], axis= 1)


def get_sentiment(node_type,node_name):
    if node_type == 0:
        return sentiment_num["sentiment"].mean()
    else:
        return sentiment_num[sentiment_num.state == name_to_state(node_name[1])]["sentiment"].mean()
# print(get_sentiment(2,['USA','Alabama','Chambers']))

