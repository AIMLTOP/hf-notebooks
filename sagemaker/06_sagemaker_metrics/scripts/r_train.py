import argparse
import logging
import os
import random
import sys

from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer

import argparse

def parse_parameters():
    parser = argparse.ArgumentParser()

    # Push to Hub Parameters
    parser.add_argument("--push_to_hub", type=bool, default=False)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_strategy", type=str, default=None)
    parser.add_argument("--hub_token", type=str, default=None)    

    args, _ = parser.parse_known_args()

    # make sure we have required parameters to push
    if args.push_to_hub:
        if args.hub_strategy is None:
            raise ValueError("--hub_strategy is required when pushing to Hub")
        if args.hub_token is None:
            raise ValueError("--hub_token is required when pushing to Hub")

    # sets hub id if not provided
    if args.hub_model_id is None:
        args.hub_model_id = args.model_id.replace("/", "--")  

    return args

def push_to_hub(trainer, hub_args):
    # save best model, metrics and create model card
    if hub_args.push_to_hub:
      trainer.create_model_card(model_name=hub_args.hub_model_id)
      trainer.push_to_hub()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=float, default=5e-5)

    # Data, model, and output directories
    parser.add_argument("--checkpoints", type=str, default="/opt/ml/checkpoints/")
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])

    args, _ = parser.parse_known_args()

    hub_args = parse_parameters()
    
    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # load datasets
    train_dataset = load_from_disk(args.training_dir)
    test_dataset = load_from_disk(args.test_dir)

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(test_dataset)}")

    # compute metrics function for binary classification
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    # download model from model hub
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # define training args
    training_args = TrainingArguments(
        output_dir=args.checkpoints,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="epoch",
        logging_dir=f"{args.checkpoints}/logs",
        learning_rate=args.learning_rate,
        # push to hub parameters
        push_to_hub=hub_args.push_to_hub,
        hub_strategy=hub_args.hub_strategy,
        hub_model_id=hub_args.hub_model_id,
        hub_token=hub_args.hub_token,
    )

    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )

    # train model
    trainer.train()

    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=test_dataset)

    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(args.checkpoints, "eval_results.txt"), "w") as writer:
        print(f"***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")

    # Saves the model locally. In SageMaker, writing in /opt/ml/model sends it to S3
    trainer.save_model(args.model_dir)
    
    push_to_hub(trainer, hub_args)
