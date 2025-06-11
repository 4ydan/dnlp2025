import requests
import stanza

def download_squad_dataset():
    url_train = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
    url_eval = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"

    response_train = requests.get(url_train)
    response_eval = requests.get(url_eval)

    if response_train.status_code != 200 or response_eval.status_code != 200:
        print("Error downloading dataset")
        print(response_eval.status_code)
    else:
        train_set = response_train.json()
        eval_set = response_eval.json()

    return train_set, eval_set
