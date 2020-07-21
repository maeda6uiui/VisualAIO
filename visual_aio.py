import argparse
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
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

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

#Set up a logger.
logging_fmt = "%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
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

def load_examples(json_filename,option_num=4,use_fixed_label=True):
    """
    Loads examples from a JSON file.

    Parameters
    ----------
    json_filename: str
        Filename of the JSON file
    option_num: int
        Number of options
    use_fixed_label: bool
        Labels are fixed to 0 if this flag is true.

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

        if use_fixed_label==True:
            label=0
        else:
            label=options.index(answer)

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
            try:
                image_pil=Image.open(image_dir+file)
            except:
                logger.error("Image file open error: {}".format(image_dir+file))
                continue

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

            #Image features
            if ending in article_dict:
                #Set token_type_ids.
                for j in range(max_seq_length-image_features_length,max_seq_length):
                    token_type_ids_tmp[j]=1

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

def train(model,train_dataset,batch_size=2,epoch_num=8,model_save_dir="."):
    """
    Trains the model.

    Parameters
    ----------
    model: transformers.BertForMultipleChoice
        BERT model
    train_dataset: torch.utils.data.TensorDataset
        Train dataset
    batch_size: int
        Batch size
    epoch_num: int
        Epoch num
    model_save_dir: str
        Directory to save the models in
    """
    logger.info("Start training.")
    logger.info("Batch size: {} Epoch num: {}".format(batch_size,epoch_num))

    #Create a dataloader.
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    #Set the model to train mode.
    model.train()

    #Set up an optimizer and a scheduler.
    lr = 5e-5
    eps = 1e-8
    logger.info("lr = {}".format(lr))
    logger.info("eps = {}".format(eps))

    optimizer = AdamW(model.parameters(), lr=lr, eps=eps)
    total_steps = len(train_dataloader) * epoch_num
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    log_interval = 100

    #Create a directory to save models in.
    os.makedirs(model_save_dir,exist_ok=True)

    for epoch in range(epoch_num):
        logger.info("========== Epoch {} / {} ==========".format(epoch + 1, epoch_num))

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t for t in batch)

            inputs=None
            if torch.cuda.is_available():
                inputs = {
                    "input_ids": batch[0].cuda(),
                    "attention_mask": batch[1].cuda(),
                    "token_type_ids": batch[2].cuda(),
                    "labels": batch[3].cuda(),
                }
            else:
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3],
                }

            # Initialize gradiants
            optimizer.zero_grad()
            # Forward propagation
            outputs = model(**inputs)
            loss = outputs[0]
            # Backward propagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update parameters
            optimizer.step()
            scheduler.step()

            model.zero_grad()

            if step % log_interval == 0:
                logger.info("Current step: {}\tLoss: {}".format(step,loss.item()))

        #Save the parameters per epoch.
        checkpoint_filename="checkpoint_{}.bin".format(epoch)
        torch.save(model.state_dict(),model_save_dir+checkpoint_filename)

    logger.info("Finished training.")
    
    #Save the final model parameters.
    torch.save(model.state_dict(), model_save_dir+"pytorch_model.bin")
    logger.info("Model saved in {}.".format(model_save_dir+"pytorch_model.bin"))

def simple_accuracy(preds, labels):
    """
    Calculates accuracy.

    Parameters
    ----------
    preds: numpy.ndarray
        Predicted labels
    labels: numpy.ndarray
        Correct labels

    Returns
    ----------
    accuracy: float
        Accuracy
    """
    return (preds == labels).mean()

def test(model,test_dataset,batch_size=4,result_filename="",labels_filename=""):
    """
    Tests the model.

    Parameters
    ----------
    model: transformers.BertForMultipleChoice
        Fine-tuned model
    test_dataset: torch.utils.data.TensorDataset
        Test dataset
    batch_size: int
        Batch size
    result_filename: str
        Filename of the text file to save the test result in.
    labels_filename: str
        Filename of the text file to save the predicted labels and the correct labels.
    """
    logger.info("Start test.")

    #Create a dataloader.
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    #Set the model to evaluation mode.
    model.eval()

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for step, batch in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():
            batch = tuple(t for t in batch)

            inputs=None
            if torch.cuda.is_available():
                inputs = {
                    "input_ids": batch[0].cuda(),
                    "attention_mask": batch[1].cuda(),
                    "token_type_ids": batch[2].cuda(),
                    "labels": batch[3].cuda(),
                }
            else:
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3],
                }

            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
            )

    eval_loss = eval_loss / nb_eval_steps
    pred_ids = np.argmax(preds, axis=1)

    accuracy = simple_accuracy(pred_ids, out_label_ids)

    #Save the test result.
    if result_filename!="":
        with open(result_filename,mode="w") as w:
            w.write("Eval loss: {}\n".format(eval_loss))
            w.write("Accuracy: {}\n".format(accuracy))

    #Save the predicted labels and correct labels.
    if labels_filename!="":
        with open(labels_filename,mode="w") as w:
            for pred,correct in zip(pred_ids,out_label_ids):
                w.write("{} {}\n".format(pred,correct))

    logger.info("Finished test.")
    logger.info("Eval loss: {}\nAccuracy: {}".format(eval_loss, accuracy))

def main(do_train=True,train_batch_size=2,train_epoch_num=5,model_save_dir="./OutputDir/"):
    """
    Main function

    Parameters
    ----------
    do_train: bool
        Runs model training if true.
    train_batch_size: int
        Batch size for model training
    train_epoch_num: int
        Number of epochs for model training
    model_save_dir: str
        Directory to save the trained model in.
    """
    IMAGE_BASE_DIR="../Data/WikipediaImages/Images/"
    ARTICLE_LIST_FILENAME="../Data/WikipediaImages/article_list.txt"

    CANDIDATE_ENTITIES_FILENAME="../Data/candidate_entities.json.gz"
    TRAIN_JSON_FILENAME="../Data/train_questions.json"
    DEV1_JSON_FILENAME="../Data/dev1_questions.json"
    DEV2_JSON_FILENAME="../Data/dev2_questions.json"
    LEADERBOARD_JSON_FILENAME="../Data/aio_leaderboard.json"

    TRAIN_OPTION_NUM=4

    TRAIN_FEATURES_CACHE_DIR="../Data/Cache/Train/"
    DEV2_FEATURES_CACHE_DIR="../Data/Cache/Dev2/"

    TEST_BATCH_SIZE=4

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
    
    #Load contexts.
    logger.info("Start loading contexts.")
    context_dict=load_contexts(CANDIDATE_ENTITIES_FILENAME)
    logger.info("Finished loading contexts.")
    logger.info("Number of contexts: {}".format(len(context_dict)))

    #Create a model.
    model = BertForMultipleChoice.from_pretrained(
        "cl-tohoku/bert-base-japanese-whole-word-masking"
    )
    if torch.cuda.is_available():
        model.cuda()

    #If there exists a cached file for the model parameters, then load it.
    model_filename=model_save_dir+"pytorch_model.bin"
    if os.path.exists(model_filename):
        logger.info("Loads parameters from {}.".format(model_filename))
        model.load_state_dict(torch.load(model_filename))

    if do_train==True:
        #Train
        train_dataset=None

        #Load cached features it cache directory exists.
        if os.path.exists(TRAIN_FEATURES_CACHE_DIR):
            logger.info("Load features from cached files.")

            input_ids=torch.load(TRAIN_FEATURES_CACHE_DIR+"input_ids.pt")
            attention_mask=torch.load(TRAIN_FEATURES_CACHE_DIR+"attention_mask.pt")
            token_type_ids=torch.load(TRAIN_FEATURES_CACHE_DIR+"token_type_ids.pt")
            labels=torch.load(TRAIN_FEATURES_CACHE_DIR+"labels.pt")

            train_dataset=torch.utils.data.TensorDataset(
                input_ids,attention_mask,token_type_ids,labels
            )

        else:
            logger.info("Start loading examples.")
            logger.info("JSON filename: {}".format(TRAIN_JSON_FILENAME))
            examples=load_examples(TRAIN_JSON_FILENAME,option_num=TRAIN_OPTION_NUM,use_fixed_label=True)
            logger.info("Finished loading examples.")
            logger.info("Number of examples: {}".format(len(examples)))

            logger.info("Start converting examples to features.")
            input_ids,attention_mask,token_type_ids,labels=convert_examples_to_features(
                examples,context_dict,article_dict,
                option_num=TRAIN_OPTION_NUM,max_seq_length=512,image_features_length=50)
            logger.info("Finished converting examples to features.")

            os.makedirs(TRAIN_FEATURES_CACHE_DIR)

            torch.save(input_ids,TRAIN_FEATURES_CACHE_DIR+"input_ids.pt")
            torch.save(attention_mask,TRAIN_FEATURES_CACHE_DIR+"attention_mask.pt")
            torch.save(token_type_ids,TRAIN_FEATURES_CACHE_DIR+"token_type_ids.pt")
            torch.save(labels,TRAIN_FEATURES_CACHE_DIR+"labels.pt")
            logger.info("Saved cache files in {}.".format(TRAIN_FEATURES_CACHE_DIR))

            train_dataset=torch.utils.data.TensorDataset(
                input_ids,attention_mask,token_type_ids,labels
            )

        train(model,train_dataset,batch_size=train_batch_size,
            epoch_num=train_epoch_num,model_save_dir=model_save_dir)
    
    #Test
    test_dataset=None

    #Load cached features it cache directory exists.
    if os.path.exists(DEV2_FEATURES_CACHE_DIR):
        logger.info("Load features from cached files.")

        input_ids=torch.load(DEV2_FEATURES_CACHE_DIR+"input_ids.pt")
        attention_mask=torch.load(DEV2_FEATURES_CACHE_DIR+"attention_mask.pt")
        token_type_ids=torch.load(DEV2_FEATURES_CACHE_DIR+"token_type_ids.pt")
        labels=torch.load(DEV2_FEATURES_CACHE_DIR+"labels.pt")

        test_dataset=torch.utils.data.TensorDataset(
            input_ids,attention_mask,token_type_ids,labels
        )

    else:
        logger.info("Start loading examples.")
        logger.info("JSON filename: {}".format(DEV2_JSON_FILENAME))
        examples=load_examples(DEV2_JSON_FILENAME,option_num=20,use_fixed_label=False)
        logger.info("Finished loading examples.")
        logger.info("Number of examples: {}".format(len(examples)))

        logger.info("Start converting examples to features.")
        input_ids,attention_mask,token_type_ids,labels=convert_examples_to_features(
            examples,context_dict,article_dict,
            option_num=20,max_seq_length=512,image_features_length=50)
        logger.info("Finished converting examples to features.")

        os.makedirs(DEV2_FEATURES_CACHE_DIR)

        torch.save(input_ids,DEV2_FEATURES_CACHE_DIR+"input_ids.pt")
        torch.save(attention_mask,DEV2_FEATURES_CACHE_DIR+"attention_mask.pt")
        torch.save(token_type_ids,DEV2_FEATURES_CACHE_DIR+"token_type_ids.pt")
        torch.save(labels,DEV2_FEATURES_CACHE_DIR+"labels.pt")
        logger.info("Saved cache files in {}.".format(DEV2_FEATURES_CACHE_DIR))

        test_dataset=torch.utils.data.TensorDataset(
            input_ids,attention_mask,token_type_ids,labels
        )

    test(model,test_dataset,batch_size=4,
        result_filename=model_save_dir+"result.txt",labels_filename=model_save_dir+"labels.txt")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="VisualAIO")

    parser.add_argument("--do_train",action="store_true")
    parser.add_argument("--train_batch_size",type=int,default=2)
    parser.add_argument("--train_epoch_num",type=int,default=5)
    parser.add_argument("--model_save_dir",type=str,default="./OutputDir/")

    args=parser.parse_args()

    main(do_train=args.do_train,
        train_batch_size=args.train_batch_size,
        train_epoch_num=args.train_epoch_num,
        model_save_dir=args.model_save_dir)
