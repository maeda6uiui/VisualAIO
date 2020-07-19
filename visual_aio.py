import gzip
import json
import logging
import os
import sys
from tqdm import tqdm
import numpy as np
import torch
from transformers import (
    BertJapaneseTokenizer,
    BertForMultipleChoice,
    AdamW,
    get_linear_schedule_with_warmup,
)

#Tokenizer
tokenizer = BertJapaneseTokenizer.from_pretrained(
    "cl-tohoku/bert-base-japanese-whole-word-masking"
)

#Set up a logger
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

class InputExample(object):
    """
    Input example
    """
    def __init__(self, qid, question, endings, label=None):
        self.qid = qid
        self.question = question
        self.endings = endings
        self.label = label

def load_examples(json_filename,option_num=4):
    """
    Loads examples from a JSON file.

    Parameters
    ----------
    json_filename: str
        Filename of the JSON file

    Returns
    ----------
    examples: InputExamples
        Examples
    """
    examples = []

    with open(json_filename, "r", encoding="UTF-8") as r:
        lines = r.read().splitlines()

    for line in lines:
        data = json.loads(line)

        qid = data["qid"]
        question = data["question"].replace("_", "")
        options = data["answer_candidates"][:option_num]
        answer = data["answer_entity"]

        #Label is always 0 for training data, and it is not necessary for test.
        label=0

        example = InputExample(qid, question, options, label)
        examples.append(example)

    return examples

def load_contexts(gz_filename):
    """
    Loads contexts from a gzip file.

    Parameters
    ----------
    gz_filename: str
        Filename of the gzip file

    Returns
    ----------
    contexts_dict: {str:str}
    """
    contexts_dict={}

    with gzip.open(gz_filename,mode="rt",encoding="utf-8") as r:
        for line in r:
            data = json.loads(line)

            title=data["title"]
            text=data["text"]

            contexts_dict[title]=text

    return contexts_dict

def main():
    CANDIDATE_ENTITIES_FILENAME="../Data/candidate_entities.json.gz"
    TRAIN_JSON_FILENAME="../Data/train_questions.json"
    DEV1_JSON_FILENAME="../Data/dev1_questions.json"
    DEV2_JSON_FILENAME="../Data/dev2_questions.json"
    
    logger.info("Start loading contexts.")
    contexts=load_contexts(CANDIDATE_ENTITIES_FILENAME)
    logger.info("Finished loading contexts.")
    logger.info("Number of contexts: {}".format(len(contexts)))

if __name__=="__main__":
    main()
