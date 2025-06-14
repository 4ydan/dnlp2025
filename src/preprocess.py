import requests
import os
from tqdm import tqdm

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

def tokenize(text, nlp):
    text = text.lower()
    doc = nlp(text)
    
    tokens = []
    for sen in doc.sentences:
        for token in sen.tokens:
            tokens.append(token.text)
    return tokens

def map_char_to_token(context, tokens):

    concat = ""
    curr = 0
    mapping = {}

    for i, char in enumerate(context):
        if char != ' ' and char != '\n':
            concat += char
            ctoken = tokens[curr]
            if concat == ctoken:
                start = i - len(concat) + 1
                for loc in range(start, i+1):
                    mapping[loc] = (concat, curr)
                concat = ""
                curr += 1
    if curr != len(tokens):
        return None
    else:
        return mapping

def process_split(split, nlp):
    mappingissues = 0
    spanissues = 0
    tokenissues = 0
    dataset = []
    for article in tqdm(split["data"], desc="Processing articles"):
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            context.replace("''",'" ') 
            context.replace("``",'" ') 
            context_tokens = tokenize(context, nlp)
            context = context.lower()

            mapping =  map_char_to_token(context, context_tokens)

            if mapping is None:
                mappingissues += 1
                print(article["title"])
                continue
            
            for qa in paragraph["qas"]:
                question_tokens = tokenize(qa["question"], nlp)

                answer_text = qa["answers"][0]["text"].lower()
                answer_start = qa["answers"][0]["answer_start"]
                answer_end = answer_start + len(answer_text)

                if context[answer_start:answer_end] != answer_text:
                    spanissues += 1
                    continue

                answer_start_wordloc = mapping[answer_start][1]
                answer_end_wordloc = mapping[answer_end-1][1]

                answer_tokens = context_tokens[answer_start_wordloc:answer_end_wordloc+1]

                if "".join(answer_tokens) != "".join(answer_text.split()):
                    tokenissues += 1
                    continue
                dataset.append((' '.join(context_tokens), ' '.join(question_tokens), ' '.join(answer_tokens), ' '.join([str(answer_start_wordloc), str(answer_end_wordloc)])))
    print(f"mappingissues: {mappingissues}")
    print(f"spanissues: {spanissues}")
    print(f"tokenissues: {mappingissues}")            
    return dataset

def write_to_files(dataset, name):
    current_dir = os.path.dirname(os.path.abspath(os.getcwd()))

    context_file_path = os.path.join(current_dir, name + ".context")
    question_file_path = os.path.join(current_dir, name + ".question")
    answer_file_path = os.path.join(current_dir, name + ".answer")
    span_file_path = os.path.join(current_dir, name + ".span")

    context_tokens = []
    question_tokens = []
    answer_tokens = []
    span_tokens = []

    with open(context_file_path, "w", encoding="utf-8") as context_f, \
        open(question_file_path, "w", encoding="utf-8") as question_f, \
        open(answer_file_path, "w", encoding="utf-8") as answer_f, \
        open(span_file_path, "w", encoding="utf-8") as span_f:
        
        for data in dataset: 
            (context, question, answer, span) = data

            context_f.write(context + "\n") 
            question_f.write(question + "\n") 
            answer_f.write(answer + "\n") 
            span_f.write(span + "\n") 

            context_tokens.append(context)
            question_tokens.append(question)
            answer_tokens.append(answer)
            span_tokens.append(span)
    return context_tokens, question_tokens, answer_tokens, span_tokens
            
