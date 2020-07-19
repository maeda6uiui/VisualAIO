import gzip
import json
import logging
import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
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

#Object detection
import cv2
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from PIL import Image

#Setup Detectron2
setup_logger()

cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)
if torch.cuda.is_available()==False:
    cfg.MODEL.DEVICE="cpu"

predictor = DefaultPredictor(cfg)

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
    option_num: int
        Number of options

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
    contexts_dict: {str: str}
    """
    contexts_dict={}

    with gzip.open(gz_filename,mode="rt",encoding="utf-8") as r:
        for line in r:
            data = json.loads(line)

            title=data["title"]
            text=data["text"]

            contexts_dict[title]=text

    return contexts_dict

def get_pred_boxes_as_images(image_dir):
        """
        Returns predicted boxes as images.

        Parameters
        ----------
        image_dir: str
            Directory of the images

        Returns
        ----------
        regions: [PIL.Image]
            Predicted boxes as images
        """
        regions=[]

        files = os.listdir(image_dir)
        for file in files:
            image_pil=Image.open(image_dir+file)

            image_cv2 = cv2.imread(image_dir + file)
            outputs = predictor(image_cv2)

            pred_boxes_tmp = outputs["instances"].pred_boxes.tensor
            for i in range(pred_boxes_tmp.size()[0]):
                top_left_x=int(pred_boxes_tmp[i][0])
                top_left_y=int(pred_boxes_tmp[i][1])
                bottom_right_x=int(pred_boxes_tmp[i][2])
                bottom_right_y=int(pred_boxes_tmp[i][3])

                image_region=image_pil.crop((top_left_x,top_left_y,bottom_right_x,bottom_right_y))
                regions.append(image_region)

        return regions

"""
def convert_examples_to_features(examples,contexts,option_num=4,max_seq_length=512,image_features_length=50):

    Converts examples to features.

    Parameters
    ----------
    examples: [InputExample]
        Input examples
    contexts: {str: str}
        Dict of contexts
    option_num: int
        Number of options
    max_seq_length: int
        Max length of input sequence to BERT
    image_features_length: int
        Length allocated for image features

    Returns
    ----------
    input_ids: torch.tensor
        Input ids for BERT
    attention_mask: torch.tensor
        Attention mask for BERT
    token_type_ids: torch.tensor
        Token type IDs for BERT
    labels: torch.tensor
        Labels for BERT
    
    input_ids=torch.empty(len(examples),option_num,max_seq_length,dtype=torch.long)
    attention_mask=torch.empty(len(examples),option_num,max_seq_length,dtype=torch.long)
    token_type_ids=torch.empty(len(examples),option_num,max_seq_length,dtype=torch.long)
    labels=torch.empty(len(examples),dtype=torch.long)

    for example in examples:
        #Process every option.
        for i,ending in enumerate(example.endings):
            text_a=example.question+"[SEP]"+ending
            text_b=contexts[ending]

            encoding = tokenizer.encode_plus(
                text_a,
                text_b,
                return_tensors="pt",
                add_special_tokens=True,
                pad_to_max_length=True,
                max_length=max_seq_length,
                truncation_strategy="only_second"   #Truncate the context
            )

            input_ids_tmp=encoding["input_ids"]
            input_ids_tmp=input_ids_tmp.view(-1)



    return input_ids,attention_mask,token_type_ids,labels
"""

def main():
    IMAGE_BASE_DIR="../Data/WikipediaImages/Images/"
    ARTICLE_LIST_FILENAME="../Data/WikipediaImages/article_list.txt"

    #Load the list of articles.
    df = pd.read_table(ARTICLE_LIST_FILENAME, header=None)

    #Make a map of articles.
    article_dict={}
    for row in df.itertuples(name=None):
        article_name = row[1]
        dir_1 = row[2]
        dir_2 = row[3]

        image_dir = IMAGE_BASE_DIR+str(dir_1) + "/" + str(dir_2) + "/"
        article_dict[article_name]=image_dir
    
    """
    CANDIDATE_ENTITIES_FILENAME="../Data/candidate_entities.json.gz"
    TRAIN_JSON_FILENAME="../Data/train_questions.json"
    DEV1_JSON_FILENAME="../Data/dev1_questions.json"
    DEV2_JSON_FILENAME="../Data/dev2_questions.json"
    
    logger.info("Start loading contexts.")
    contexts=load_contexts(CANDIDATE_ENTITIES_FILENAME)
    logger.info("Finished loading contexts.")
    logger.info("Number of contexts: {}".format(len(contexts)))
    """


if __name__=="__main__":
    main()
