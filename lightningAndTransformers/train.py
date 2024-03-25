from transformers import BertTokenizer, VisualBertForPreTraining
from transformers import BertModel, VisualBertModel, VisualBertConfig, VisualBertForQuestionAnswering 
from transformers import LxmertForQuestionAnswering 
from transformers import ViltForQuestionAnswering, ViltProcessor
from myDataset import *
from myViltDataset import *

import argparse
import json
from multiprocessing import reduction
from typing import Any, Dict
import numpy as np
import os
import torch
from torch import optim
from torch.utils import data
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cloud_io import load as pl_load
import transformers

from pathlib import Path
import pickle
import torchvision

from constants import *
from captions import *
from transformers import BertTokenizer
from eneko import utilities
from myDataset import *
from VisualBertForPreTrainingNew import *

class myLitModel(pl.LightningModule):
    
    def __init__(self, args):
        super().__init__()
        #create args component for later use
        self.args=args
        # Load model, tokenizer and loss function
        # MODEL
        if(args.onlyTest):
            if(args.modelName=="visualbert"):
                self.model=VisualBertForQuestionAnswering.from_pretrained("uclanlp/visualbert-vqa")
            if(args.modelName=="lxmert"):
                self.model=LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-vqa-uncased")
            if(args.modelName=="vilt"):
                self.model=ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        else:
            bert = BertModel.from_pretrained("bert-base-uncased")
            visualbert_config = VisualBertConfig.from_pretrained("uclanlp/visualbert-vqa")
            self.model = VisualBertForQuestionAnsweringNew(visualbert_config, args)
            if(not args.random):
                self.model.visual_bert.load_state_dict(bert.state_dict(), strict=False)
        # print(self.model)
        # print("selected weigths:")
        # print(self.model.base_model.encoder.layer[2]._modules["intermediate"].dense.weight)
        # TOKENIZER
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # LOSS
        self.loss = torch.nn.BCEWithLogitsLoss()
        # LOAD LABELS
        self.task_name = args.dataset
        data = self.task_name.split("_")
        self.dataset = data[0]
        self.dataset_version = data[1]

        if(args.onlyTest):
            if(args.modelName=="visualbert"):
                with open(os.path.join("data", f"{self.dataset}_answer_vocab_idx2word_{self.dataset_version}_new.json"), "r") as f:
                    self.labels = json.load(f)
                    #self.labels.append("UNK")
                    self.label_ids= {l: i for i, l in enumerate(self.labels)}
                    self.num_labels = len(self.labels)
            if(args.modelName=="lxmert"):
                with open(os.path.join("data", f"lxmertLabel2Ans.json"), "r") as f:
                    self.labels = json.load(f)
                    #self.labels.append("UNK")
                    self.label_ids= {l: i for i, l in enumerate(self.labels)}
                    self.num_labels = len(self.labels)
            # ETIKETAK PROGRAMATZEKO DAUDE ORAINDIK
            if(args.modelName=="vilt"):
                self.labels = list(self.model.config.id2label.values())
                #self.labels.append("UNK")
                self.label_ids = {l: i for i, l in enumerate(self.labels)}
                self.num_labels = len(self.labels)
        else:
            with open(os.path.join("data", f"{self.dataset}_answer_vocab_idx2word_{self.dataset_version}.json"), "r") as f:
                self.labels = json.load(f)
                self.labels.append("UNK")
                self.label_ids = {l: i for i, l in enumerate(self.labels)}
                self.num_labels = len(self.labels)

        # Define other hyperparameters
        self.warmup_steps = args.warmup_steps
        self.max_steps = args.max_steps
        self.lr = args.lr
        self.opt_eps = args.opt_eps
        self.opt_wd = args.opt_wd

        self.pretrained_on = None
        self.prev_num_labels = 0
        

    def forward(self, batch):
        
        if(not self.args.modelName=="vilt"):
            features, question, answers, positionEncodingOriginal, positionEncodingGrid = batch
        if(self.args.modelName == "vilt"):
            viltInput, answers = batch
            
        # ENEKO: Tokenize and create the necessary embeddings
        if(not self.args.modelName=="vilt"):
            tokens = self.tokenizer(question, return_tensors="pt", padding="max_length", max_length=50)
            input_ids = tokens["input_ids"]
            input_ids = input_ids.to(self.device)
            token_type_ids = tokens["token_type_ids"]
            token_type_ids = token_type_ids.to(self.device)
            attention_mask = tokens["attention_mask"]
            attention_mask = attention_mask.to(self.device)


        if(self.args.modelName=="visualbert"):
            # Forward pass
            logits = self.model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                visual_embeds=features,
                positionEncodingOriginal=positionEncodingOriginal,
                positionEncodingGrid=positionEncodingGrid,
                attention_mask=attention_mask
            )
        
        if(self.args.modelName=="lxmert"):
            # Forward pass
            logits = self.model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                visual_feats=features,
                visual_pos=positionEncodingOriginal,
                attention_mask=attention_mask
            )

        if(self.args.modelName=="vilt"):
            # Forward pass
            logits = self.model(
                input_ids=viltInput["input_ids"].squeeze(1),
                token_type_ids=viltInput["token_type_ids"].squeeze(1),
                pixel_values=viltInput["pixel_values"].squeeze(1),
                pixel_mask=viltInput["pixel_mask"].squeeze(1),
                attention_mask=viltInput["attention_mask"].squeeze(1)
            )
        return logits

    def configure_optimizers(self):
        # Define optimizer and scheduler
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, eps=self.opt_eps, weight_decay=self.opt_wd)
        scheduler = {
            "scheduler": transformers.get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.max_steps),
            "interval": "step"
        }
        return [optimizer], [scheduler]
    
    def general_step(self, batch, split="train"):
        if(not self.args.modelName=="vilt"):
            features, question, answers, positionEncodingOriginal, positionEncodingGrid = batch
        if(self.args.modelName == "vilt"):
            viltInput, answers = batch
        #tokens = tokenizer(answers, return_tensors="pt", padding="max_length", max_length=2048)
        #answer_ids = tokens["input_ids"]
        # Model forward pass
        output = self(batch)

        # MOVE: Load target data 
        # answers = batch[-1]
        if(not self.args.modelName=="vilt"):
            transposed_answers = list(map(list, zip(*answers)))
        else:
            transposed_answers=answers
        target = utilities.build_target_tensor(transposed_answers, self.label_ids).to(self.device)    

        if(self.args.modelName=="visualbert"):
            loss = self.loss(output.logits, target)
        if(self.args.modelName=="lxmert"):
            loss = self.loss(output.question_answering_score, target)
        if(self.args.modelName=="vilt"):
            loss = self.loss(output.logits, target)

        # Compute Accuracy (don't take UNK token into account)
        if(self.args.modelName=="visualbert"):
            _, indices = output.logits.topk(2)
        if(self.args.modelName=="lxmert"):
            _, indices = output.question_answering_score.topk(2)
        if(self.args.modelName=="vilt"):
            _, indices = output.logits.topk(2)
        predictions = [
            int(ids[0].data.cpu().numpy())
            #if ids[0] < len(self.labels) - 1
            #else int(ids[1].data.cpu().numpy())
            for ids in indices
        ]
        accuracy = np.mean(
            np.array(
                [utilities.vqa_accuracy(ans, self.labels[int(p_id)]) for ans, p_id in zip(transposed_answers, predictions)]
            )
        ) 
        #with open(1, "w", encoding='utf-8', closefd=False) as f:
        #    print(batch, file=f)
        #    print([self.labels[int(p_id)] for p_id in predictions], file=f)
        #exit(0)
        # Save metrics
        self.log(f'{split}_loss', loss, on_epoch=True, prog_bar=(split=="train"), logger=True)
        self.log(f'{split}_accuracy', accuracy, on_epoch=True, prog_bar=(split=="train"), logger=True)

        return loss

    def predict_step(self, batch, split="train"):
        if(not self.args.modelName=="vilt"):
            features, question, answers, positionEncodingOriginal, positionEncodingGrid = batch
        if(self.args.modelName == "vilt"):
            viltInput, answers = batch
        #tokens = tokenizer(answers, return_tensors="pt", padding="max_length", max_length=2048)
        #answer_ids = tokens["input_ids"]
        # Model forward pass
        output = self(batch)

        transposed_answers = list(map(list, zip(*answers)))

        target = utilities.build_target_tensor(transposed_answers, self.label_ids).to(self.device)    



        # Compute Accuracy (don't take UNK token into account)
        if(self.args.modelName=="visualbert"):
            _, indices = output.logits.topk(2)
        if(self.args.modelName=="lxmert"):
            _, indices = output.question_answering_score.topk(2)
        if(self.args.modelName=="vilt"):
            _, indices = output.logits.topk(2)
        predictions = [
            int(ids[0].data.cpu().numpy())
            #if ids[0] < len(self.labels) - 1
            #else int(ids[1].data.cpu().numpy())
            for ids in indices
        ]
        print("predictions")
        print([self.labels[int(p_id)] for p_id in predictions])
        print("answers")
        print([p_id for p_id in transposed_answers])
        return predictions

    def training_step(self, batch, batch_idx):
        return self.general_step(batch, split="train")
    
    def validation_step(self, batch, batch_idx):
        return self.general_step(batch, split="val")

    def test_step(self, batch, batch_idx):
        return self.general_step(batch, split="test")

class myDataloader(pl.LightningDataModule):

    def __init__(self, args, grid_size=16):
        super().__init__()
        
        self.feat_path = VQA_FEAT_PATH
        self.is_tiny =args.tiny

        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        # ENEKO: new arg grid_size
        self.grid_size = args.grid_size
        self.is_vsr = args.is_vsr
        self.modelName = args.modelName

    # TODO: Download everything and structure it
    """
    def prepare_data(self):
        # Download everything
        raise NotImplementedError
    """

    def train_dataloader(self):
        if (not self.modelName=="vilt"):
            dataset =  myDataset(VQA_ROOT, VQA_TRAIN_QUESTION_FILE, ann_file=VQA_TRAIN_ANN_FILE, feat_path=VQA_FEAT_PATH,
                partition = "train", grid_size=self.grid_size, is_tiny=self.is_tiny, is_vsr=self.is_vsr)
        else:
            dataset =  myViltDataset(VQA_ROOT, VQA_TRAIN_QUESTION_FILE, ann_file=VQA_TRAIN_ANN_FILE, feat_path=VQA_FEAT_PATH,
                partition = "train", grid_size=self.grid_size, is_tiny=self.is_tiny, is_vsr=self.is_vsr)
        params = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': self.num_workers
        }
        if(self.modelName=="vilt"):
            params["collate_fn"]=dataset.collate_fn
        return data.DataLoader(dataset, **params)
    
    def val_dataloader(self):
        if (not self.modelName=="vilt"):
            dataset = myDataset(VQA_ROOT, VQA_VAL_QUESTION_FILE, ann_file=VQA_VAL_ANN_FILE, feat_path=VQA_FEAT_PATH,
                partition = "val", grid_size=self.grid_size, is_tiny=self.is_tiny, is_vsr=self.is_vsr)
        else:
            dataset = myViltDataset(VQA_ROOT, VQA_VAL_QUESTION_FILE, ann_file=VQA_VAL_ANN_FILE, feat_path=VQA_FEAT_PATH,
                partition = "val", grid_size=self.grid_size, is_tiny=self.is_tiny, is_vsr=self.is_vsr)
        params = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers
        }
        if(self.modelName=="vilt"):
            params["collate_fn"]=dataset.collate_fn
        return data.DataLoader(dataset, **params)

    def test_dataloader(self):
        if (not self.modelName=="vilt"):
            dataset = myDataset(VQA_ROOT, VQA_VAL_QUESTION_FILE, ann_file=VQA_VAL_ANN_FILE, feat_path=VQA_FEAT_PATH,
                partition = "test", grid_size=self.grid_size, is_tiny=self.is_tiny, is_vsr=self.is_vsr)
        else:
            dataset = myViltDataset(VQA_ROOT, VQA_VAL_QUESTION_FILE, ann_file=VQA_VAL_ANN_FILE, feat_path=VQA_FEAT_PATH,
                partition = "test", grid_size=self.grid_size, is_tiny=self.is_tiny, is_vsr=self.is_vsr)
        params = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers
        }
        if(self.modelName=="vilt"):
            params["collate_fn"]=dataset.collate_fn
        return data.DataLoader(dataset, **params)

    def predict_dataloader(self):
        if (not self.modelName=="vilt"):
            dataset = myDataset(VQA_ROOT, VQA_VAL_QUESTION_FILE, ann_file=VQA_VAL_ANN_FILE, feat_path=VQA_FEAT_PATH,
                partition = "val", grid_size=self.grid_size, is_tiny=self.is_tiny, is_vsr=self.is_vsr)
        else:
            dataset = myViltDataset(VQA_ROOT, VQA_VAL_QUESTION_FILE, ann_file=VQA_VAL_ANN_FILE, feat_path=VQA_FEAT_PATH,
                partition = "val", grid_size=self.grid_size, is_tiny=self.is_tiny, is_vsr=self.is_vsr)
        params = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers
        }
        if(self.modelName=="vilt"):
            params["collate_fn"]=dataset.collate_fn
        return data.DataLoader(dataset, **params)

def parse_args():
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--dataset", type=str, default="okvqa_v1.1", choices=["okvqa_v1.0", "okvqa_v1.1", "vqa_v2"],
        help="Select dataset to be trained on."
    )
    parser.add_argument(
        "--gpus", type=int, default=1, help="Number of GPUs in use. (0 == cpu)"
    )
    #METHOD: position embedding
    parser.add_argument(
        "--method", type=str, default="original", choices=["original", "empty", "grid"],
        help="The method used to codify the position embedding."
    )
    parser.add_argument(
        "--modelName", type=str, default="visualbert", choices=["visualbert", "lxmert", "vilt"],
        help="The model used for testing."
    )
    #GRID-SIZE: used only if the grid codification is chosen
    parser.add_argument(
        "--grid_size", type=int, default=32, help="Grid size (32, 28, 16)."
    )
    parser.add_argument(
        "--batch_size", type=int, default=56, help="Batch size (per gpu)."
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="Learning rate."
    )
    parser.add_argument(
        "--opt_eps", type=float, default=1e-8, help="Epsilon value for AdamW optimizer."
    )
    parser.add_argument(
        "--opt_wd", type=float, default=0.0, help="Weight decay value for AdamW optimizer."
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=2000, help="Warmup steps to be done during training."
    )
    parser.add_argument(
        "--max_steps", type=int, default=88000, help="Steps to be done during training."
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="Seed."
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Test model after fine-tuning."
    )
    parser.add_argument(
        "--run_name", type=str, default=None, help="Name of the run. Used in tensorboard and output filenames. If it is not filled or already exists, a custom one will be generated."
    )
    parser.add_argument(
        "--output_path", type=str, default="./output", help="Output directory for plots and models."
    )
    parser.add_argument(
        "--num_workers", type=int, default=12, help="Workers used in the dataloader."
    )
    parser.add_argument(
        "--tiny", action="store_true", help="Use tiny version of model, for tiny purposes..."
    )
    parser.add_argument(
        "--random", action="store_true", help="Use random initialization"
    )
    parser.add_argument(
        "--onlyTest", action="store_true", help="Only testing purposes"
    )
    parser.add_argument(
        "--is_vsr", action="store_true", help="Use vsr subset"
    )
    parser.add_argument(
        "--predict", action="store_true", help="Use vsr subset"
    )
    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    
    # Reproducibility
    if args.seed != -1:
        pl.utilities.seed.seed_everything(args.seed)

    # Load model

    # Define checkpoint filename and tensorboard run name

    ckpt_filename = str(args.run_name)
    tb_run_name = ckpt_filename + "_tb"

    print("Loading model...")
    # needed args: warmup_steps, max_steps, lr, opt_eps, opt_wd

    model = myLitModel(args)
    # Load data
    print("Loading dataset...")
    # needed args: feat_path, is_tiny, batch_size, num_workers, grid_size
    datamodule = myDataloader(args)

    # Define trainer
    logger = TensorBoardLogger("logs", name=tb_run_name, default_hp_metric=False)
    # needed args: gpus, fast_dev_run, logger, max_steps
    trainer = pl.Trainer(checkpoint_callback=False, gpus=args.gpus, fast_dev_run=args.tiny, logger=logger, max_steps=args.max_steps)

    if(not args.onlyTest):
        # Train model
        print("Training starts!")
        trainer.fit(model, datamodule)
        print("Training finished!")
        #trainer.save_checkpoint(os.path.join(args.output_path, ckpt_filename))

    # Evaluate model
    if args.evaluate:
        trainer.test(model=model, datamodule=datamodule)

    if args.predict:
        trainer.predict(model=model, datamodule=datamodule)

if __name__ == "__main__":
    main()

# ORIGINAL POSITION ENCODING (BIDALITA) ok
# screen -S screen_eneko_atxa
# setenv CUDA_VISIBLE_DEVICES 1
# source /var/python3envs/transformers-4.15.0/bin/activate.csh
# screen -S screen_eneko_atxa_tb
# tensorboard --logdir=logs/ --bind_all
# python3 myTrain.py --is_vsr --gpus 1 --method "original" --batch_size 56 --seed 42 --run_name "OC_sub" --output_path "./OC_sub" --dataset "vqa_v2" --evaluate > OC_sub.out
# python3 myTrain.py --is_vsr --gpus 1 --method "original" --batch_size 56 --seed 42 --run_name "OC_sub_not" --output_path "./OC_sub_not" --dataset "vqa_v2" --evaluate > OC_sub_not.out
# GRID POSITION ENCODING 32 (BIDALITA)
# screen -S screen_eneko_atxa
# setenv CUDA_VISIBLE_DEVICES 1
# source /var/python3envs/transformers-4.15.0/bin/activate.csh
# screen -S screen_eneko_atxa_tb
# tensorboard --logdir=logs/ --bind_all
# python3 myTrain.py --is_vsr --gpus 1 --method "grid" --grid_size 32 --batch_size 56 --seed 42 --run_name "GC32_sub" --output_path "./GC32_sub" --dataset "vqa_v2" --evaluate > GC32_sub.out

# GRID POSITION ENCODING 16 (BIDALITA) ok
# screen -S screen_eneko_atxa
# setenv CUDA_VISIBLE_DEVICES 1
# source /var/python3envs/transformers-4.15.0/bin/activate.csh
# screen -S screen_eneko_atxa_tb
# tensorboard --logdir=logs/ --bind_all
# python3 myTrain.py --is_vsr --gpus 1 --method "grid" --grid_size 16 --batch_size 56 --seed 42 --run_name "GC16_sub" --output_path "./GC16_sub" --dataset "vqa_v2" --evaluate > GC16_sub.out

# GRID POSITION ENCODING 28 (BIDALITA) ok
# screen -S screen_eneko_atxa
# setenv CUDA_VISIBLE_DEVICES 1
# source /var/python3envs/transformers-4.15.0/bin/activate.csh
# screen -S screen_eneko_atxa_tb
# tensorboard --logdir=logs/ --bind_all
# python3 myTrain.py --is_vsr --gpus 1 --method "grid" --grid_size 28 --batch_size 56 --seed 42 --run_name "GC28_sub" --output_path "./GC28_sub" --dataset "vqa_v2" --evaluate > GC28_sub.out

# GRID POSITION ENCODING 64 (BIDALITA)
# screen -S screen_eneko_atxa
# setenv CUDA_VISIBLE_DEVICES 1
# source /var/python3envs/transformers-4.15.0/bin/activate.csh
# screen -S screen_eneko_atxa_tb
# tensorboard --logdir=logs/ --bind_all
# python3 myTrain.py --is_vsr --gpus 1 --method "grid" --grid_size 64 --batch_size 56 --seed 42 --run_name "GC64_sub" --output_path "./GC64_sub" --dataset "vqa_v2" --evaluate > GC64_sub.out

# NO POSITION ENCODING (BIDALITA) ok
# screen -S screen_eneko_atxa
# setenv CUDA_VISIBLE_DEVICES 1
# source /var/python3envs/transformers-4.15.0/bin/activate.csh
# screen -S screen_eneko_atxa_tb
# tensorboard --logdir=logs/ --bind_all
# python3 myTrain.py --is_vsr --gpus 1 --method "empty" --batch_size 56 --seed 42 --run_name "EC_sub" --output_path "./EC_sub" --dataset "vqa_v2" --evaluate > EC_sub.out
# inputen tamainak esperotakoak izatea. Proiektatu aurretiko tamainak. Outputen tamainak. Ea if eko zeinetan sartzen den. Atentzio maskaren tamainak. Kargatutako pisuak. (print self.model. Layer bat aukeratu eta weights eta bias bat aukeratu eta atera)
# komandoaren ondoren > output_01.log

# ORIGINAL ENCODING + RANDOM WEIGHTS
# screen -S screen_eneko_atxa
# setenv CUDA_VISIBLE_DEVICES 1
# source /var/python3envs/transformers-4.15.0/bin/activate.csh
# screen -S screen_eneko_atxa_tb
# tensorboard --logdir=logs/ --bind_all
# python3 myTrain.py --gpus 1 --random --method "original" --batch_size 56 --seed 42 --run_name "OC_random" --output_path "./OC_random" --dataset "vqa_v2" --evaluate > OC_random.out

# LXMERT execution
# python3 myTrain.py --gpus 1 --method "original" --batch_size 56 --seed 42 --run_name "lxmert" --output_path "./lxmert" --dataset "vqa_v2" --evaluate --onlyTest --modelName "lxmert" > lxmert.out
# python3 myTrain.py --gpus 1 --method "original" --batch_size 56 --seed 42 --run_name "lxmert_sub" --output_path "./lxmert_sub" --dataset "vqa_v2" --evaluate --onlyTest --is_vsr --modelName "lxmert" > lxmert_sub.out

# VILT execution
# python3 myTrain.py --gpus 1 --method "original" --batch_size 56 --seed 42 --run_name "vilt" --output_path "./vilt" --dataset "vqa_v2" --evaluate --onlyTest --modelName "vilt" > vilt.out
# python3 myTrain.py --gpus 1 --method "original" --batch_size 56 --seed 42 --run_name "vilt_sub" --output_path "./vilt_sub" --dataset "vqa_v2" --evaluate --onlyTest --is_vsr --modelName "vilt" > vilt_sub.out

# source /gaueko0/users/eatxa002/myEnvironment/envVilt/bin/activate.csh

# python3 myTrain.py --gpus 1 --method "original" --batch_size 56 --seed 42 --run_name "orig" --output_path "./orig" --dataset "vqa_v2" > orig.out
# python3 myTrain.py --gpus 1 --method "empty" --batch_size 56 --seed 42 --run_name "empty" --output_path "./empty" --dataset "vqa_v2" > empty.out
# python3 myTrain.py --gpus 1 --method "grid" --grid_size 32 --batch_size 56 --seed 42 --run_name "grid32" --output_path "./grid32" --dataset "vqa_v2" > grid32.out
# python3 myTrain.py --gpus 1 --method "grid" --grid_size 64 --batch_size 56 --seed 42 --run_name "grid64" --output_path "./grid64" --dataset "vqa_v2" > grid64.out
# python3 myTrain.py --gpus 1 --method "original" --batch_size 56 --seed 42 --run_name "orig_coco" --output_path "./orig_coco" --dataset "vqa_v2" > orig_coco.out
# python3 myTrain.py --gpus 1 --method "empty" --batch_size 56 --seed 42 --run_name "empty_coco" --output_path "./empty_coco" --dataset "vqa_v2" > empty_coco.out
# python3 myTrain.py --gpus 1 --method "grid" --grid_size 32 --batch_size 56 --seed 42 --run_name "grid32_coco" --output_path "./grid32_coco" --dataset "vqa_v2" > grid32_coco.out
# python3 myTrain.py --gpus 1 --method "grid" --grid_size 64 --batch_size 56 --seed 42 --run_name "grid64_coco" --output_path "./grid64_coco" --dataset "vqa_v2" > grid64_coco.out

# kargatu hugginface, eta testeatu bakarrik (vqa eta vqa coco)
# esperimentuak egin pisuak aleatorioki hasieratuta
# lr 2e-5 ekin, berten pisuak eta OC rekin egin (EGINDA)
# excela egin, tensroboardeko libnka jarri dokumentuan overleafeko esteka ere jarri


# ViLT eta LXMERT en gainean probak egin (posizioaren kodifikazioa berak erabiltzen duena soilik erabili)
# beraiek aurreentrenatuta daude posizioa erabiltzeko berez, beraz emaitza hobeak lortu behar lituzkete.

# kontuan izan spatial subsetean ez dutela zertan galdera espazialak bakarrik egon. (kartela gezia ezkerretara)
# multilabel classification (hainbat erantzun egoki)n BCE egiten da output bakoitzaren gainean

# python3 myTrain.py --gpus 1 --method "original" --batch_size 56 --seed 42 --run_name "OC_sub" --output_path "OC_sub" --dataset "vqa_v2" --evaluate --predict > ocpredict.out
# python3 myTrain.py --gpus 1 --method "empty" --batch_size 56 --seed 42 --run_name "EC_sub" --output_path "EC_sub" --dataset "vqa_v2" --evaluate --predict > ecpredict.out
# python3 myTrain.py --gpus 1 --method "grid" --grid_size 16 --batch_size 56 --seed 42 --run_name "GC_16_sub" --output_path "GC_16_sub" --dataset "vqa_v2" --evaluate --predict > gcpredict.out

# python3 myTrain.py --gpus 1 --method "original" --batch_size 56 --seed 42 --run_name "OC_sub" --output_path "OC_sub" --dataset "vqa_v2" --evaluate --predict --is_vsr > ocvsr.out
# python3 myTrain.py --gpus 1 --method "empty" --batch_size 56 --seed 42 --run_name "EC_sub" --output_path "EC_sub" --dataset "vqa_v2" --evaluate --predict --is_vsr > ecvsr.out
# python3 myTrain.py --gpus 1 --method "grid" --grid_size 16 --batch_size 56 --seed 42 --run_name "GC_16_sub" --output_path "GC_16_sub" --dataset "vqa_v2" --evaluate --predict --is_vsr > gcvsr.out