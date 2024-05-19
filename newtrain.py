import random
from data import ImageDetectionsField, TextField, RawField
from data import DataLoader  # Assuming COCO_VQA is a modified dataset class for VQA
import evaluation
from evaluation import PTBTokenizer, Cider
from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import CrossEntropyLoss  # Using CrossEntropyLoss for classification tasks
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse, os, pickle
import numpy as np
import itertools
import multiprocessing
from shutil import copyfile
from data.dataset import OpenViVQA, Example

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)


def evaluate_loss(model, dataloader, loss_fn, text_field):
    # Validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Validation', unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, (detections, questions, answers) in enumerate(dataloader):
                detections, questions, answers = detections.to(device), questions.to(device), answers.to(device)
                out = model(detections, questions)
                loss = loss_fn(out, answers)
                running_loss += loss.item()

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    return val_loss


def evaluate_metrics(model, dataloader, text_field):
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, (images, questions, answers_gt) in enumerate(iter(dataloader)):
            images, questions = images.to(device), questions.to(device)
            with torch.no_grad():
                out = model(images, questions)
            gen[it] = out.argmax(dim=1).cpu().numpy().tolist()
            gts[it] = answers_gt.cpu().numpy().tolist()
            pbar.update()

    # Compute metrics using the evaluation module
    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    cider = evaluation.Cider()
    score, _ = cider.compute_score(gts, gen)
    return score


def scst_train_step(model, data, optimizer, cider, tokenizer, vocab):
    model.train()
    images, questions, answers = data
    images, questions, answers = images.to(device), questions.to(device), answers.to(device)

    optimizer.zero_grad()
    out = model(images, questions)
    loss = cider.compute_loss(out, answers)
    loss.backward()
    optimizer.step()
    return loss.item()


def save_checkpoint(model, optimizer, epoch, val_loss, accuracy, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'accuracy': accuracy,
    }, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--img_root', type=str, default='data/images')
    parser.add_argument('--ann_root', type=str, default='data/annotations')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fields
    image_field = ImageDetectionsField(detections_path=args.img_root, max_detections=50, load_in_tmp=False)
    question_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy')
    answer_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy')

    # Dataset
    data = OpenViVQA(image_field, question_field, answer_field, args.img_root, args.ann_root)
    train_dataset,val_dataset,test_dataset=data.splits()

    # Dataloader
    dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    dataloader_test = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # Model
    encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory, attention_module_kwargs={'m': 40})
    decoder = MeshedDecoder(len(answer_field.vocab), 54, 3, attention_module=ScaledDotProductAttentionMemory, attention_module_kwargs={'m': 40})
    model = Transformer(encoder, decoder).to(device)

    # Loss and optimizer
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.0001)
    scheduler = LambdaLR(optimizer, lambda epoch: 0.95 ** epoch)

    # Cider and tokenizer for SCST
    cider = Cider()
    tokenizer = PTBTokenizer()

    # Training loop
    for e in range(20):
        model.train()
        running_loss = 0.0
        scst_loss = 0.0
        with tqdm(desc='Epoch %d - Training' % e, unit='it', total=len(dataloader_train)) as pbar:
            for it, data in enumerate(dataloader_train):
                # SCST training step
                scst_loss += scst_train_step(model, data, optimizer, cider, tokenizer, answer_field.vocab)

                pbar.set_postfix(scst_loss=scst_loss / (it + 1))
                pbar.update()

        val_loss = evaluate_loss(model, dataloader_val, loss_fn, answer_field)
        accuracy = evaluate_metrics(model, dataloader_val, answer_field)
        scheduler.step()

        print(f'Epoch {e}, Validation Loss: {val_loss}, Accuracy: {accuracy}, SCST Loss: {scst_loss / len(dataloader_train)}')

        # Save checkpoint at the end of each epoch
        save_checkpoint(model, optimizer, e, val_loss, accuracy, path=f'checkpoint_epoch_{e}.pth')
