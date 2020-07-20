import gzip
import json
import logging
import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torchvision
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

#VGG-16
vgg16=torchvision.models.vgg16(pretrained=True)
vgg16.eval()

normalize = torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize
])

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

def get_vgg16_output_from_region(region,out_dim=1):
    """
    Returns output from VGG-16.

    Parameters
    ----------
    region: PIL.Image
        Region of an image
    out_dim: int
        Dimension of the output vector

    Returns
    ----------
    ret: torch.tensor
        Output from VGG-16
    """
    vgg16.classifier[6]=torch.nn.Linear(4096,out_dim)

    region=region.convert("RGB")
    region_tensor = preprocess(region)
    region_tensor=region_tensor.unsqueeze(0)

    ret=vgg16(region_tensor)
    ret=ret.flatten()

    return ret

def get_vgg16_output_from_regions(regions,output_dim=1):
    """
    Returns output from VGG-16.

    Parameters
    ----------
    regions: [PIL.Image]
        Regions of images
    out_dim: int
        Dimension of each output vector

    Returns
    ----------
    ret: torch.tensor
        Output from VGG-16
    """
    ret=torch.empty(0,dtype=torch.float)

    for region in regions:
        tmp=get_vgg16_output_from_region(region,output_dim)
        ret=torch.cat([ret,tmp],dim=0)

    return ret

def convert_examples_to_features(
    examples,context_dict,article_dict,
    option_num=4,max_seq_length=512,image_features_length=50):
    """
    Converts examples to features.

    Parameters
    ----------
    examples: [InputExample]
        Input examples
    context_dict: {str: str}
        Dict of contexts
    article_dict: {str: str}
        Dict of article names and image directories
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
    """
    input_ids=torch.empty(len(examples),option_num,max_seq_length,dtype=torch.long)
    attention_mask=torch.empty(len(examples),option_num,max_seq_length,dtype=torch.long)
    token_type_ids=torch.empty(len(examples),option_num,max_seq_length,dtype=torch.long)
    labels=torch.empty(len(examples),dtype=torch.long)

    for example_index,example in enumerate(tqdm(examples)):
        #Process every option.
        for i,ending in enumerate(example.endings):
            #Text features
            text_a=example.question+"[SEP]"+ending
            text_b=context_dict[ending]

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

            #1 for real tokens and 0 for padding
            attention_mask_tmp=torch.ones(max_seq_length,dtype=torch.long)
            #0 for text features and 1 for image features
            token_type_ids_tmp=torch.zeros(max_seq_length,dtype=torch.long)
            for j in range(max_seq_length-image_features_length,max_seq_length):
                token_type_ids_tmp[j]=1

            #Image features
            if ending in article_dict:
                image_dir=article_dict[ending]
                regions=get_pred_boxes_as_images(image_dir)
                image_features=get_vgg16_output_from_regions(regions,output_dim=1)

                #Some transformation to have this features as an input to transformers' BERT model. 
                OFFSET=1.0
                SCALE=10000

                image_features=(image_features+OFFSET)*SCALE
                image_features=image_features.long()
                image_features=torch.clamp(image_features,0,len(tokenizer))

                #Make room for the image features.
                input_ids_tmp=input_ids_tmp[:max_seq_length-image_features_length]
                input_ids_tmp=torch.cat([input_ids_tmp,image_features],dim=0)

                input_ids_length=input_ids_tmp.size()[0]

                #Truncate input_ids if it is too long.
                if input_ids_length>max_seq_length:
                    input_ids_tmp=input_ids_tmp[:max_seq_length]
                #Pad with zero if it is too short.
                elif input_ids_length<max_seq_length:
                    pad_length=max_seq_length-input_ids_length
                    zero_pad=torch.zeros(pad_length,dtype=torch.long)

                    input_ids_tmp=torch.cat([input_ids_tmp,zero_pad],dim=0)

                    #Set attention mask.
                    for j in range(input_ids_length,max_seq_length):
                        attention_mask_tmp[j]=0

            input_ids[example_index,i]=input_ids_tmp
            token_type_ids[example_index,i]=token_type_ids_tmp
            attention_mask[example_index,i]=attention_mask_tmp

        labels[example_index]=example.label

    return input_ids,attention_mask,token_type_ids,labels

def main():
    IMAGE_BASE_DIR="../Data/WikipediaImages/Images/"
    ARTICLE_LIST_FILENAME="../Data/WikipediaImages/article_list.txt"

    #Load the list of articles.
    logger.info("Start loading the article list.")
    df = pd.read_table(ARTICLE_LIST_FILENAME, header=None)
    logger.info("Finished loading the article list.")

    #Make a dict of articles.
    logger.info("Start creating a dict of articles.")

    article_dict={}
    for row in df.itertuples(name=None):
        article_name = row[1]
        dir_1 = row[2]
        dir_2 = row[3]

        image_dir = IMAGE_BASE_DIR+str(dir_1) + "/" + str(dir_2) + "/"
        article_dict[article_name]=image_dir

    logger.info("Finished creating a dict of articles.")

    CANDIDATE_ENTITIES_FILENAME="../Data/candidate_entities.json.gz"
    TRAIN_JSON_FILENAME="../Data/train_questions.json"
    DEV1_JSON_FILENAME="../Data/dev1_questions.json"
    DEV2_JSON_FILENAME="../Data/dev2_questions.json"
    
    logger.info("Start loading contexts.")
    context_dict=load_contexts(CANDIDATE_ENTITIES_FILENAME)
    logger.info("Finished loading contexts.")
    logger.info("Number of contexts: {}".format(len(context_dict)))

    logger.info("Start loading examples.")
    examples=load_examples(DEV2_JSON_FILENAME,option_num=4)
    logger.info("Finished loading examples.")
    logger.info("Number of examples: {}".format(len(examples)))

    logger.info("Start converting examples to features.")
    input_ids,attention_mask,token_type_ids,labels=convert_examples_to_features(
        examples,context_dict,article_dict,
        option_num=4,max_seq_length=512,image_features_length=50)
    logger.info("Finished converting examples to features.")

    print("input_ids[0]:",input_ids.detach().numpy()[0])
    print("attention_mask[0]:",attention_mask.detach().numpy()[0])
    print("token_type_ids[0]:",token_type_ids.detach().numpy()[0])
    print("labels:",labels.detach().numpy())

if __name__=="__main__":
    main()
