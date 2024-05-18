import random
from data import ImageDetectionsField, TextField, RawField
from data import COCO_VQA, DataLoader  # Assuming COCO_VQA is a modified dataset class for VQA
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
from data.dataset import OpenViVQA
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

    # Assuming we have a function `vqa_accuracy` to calculate accuracy for VQA
    accuracy = vqa_accuracy(gen, gts)
    return accuracy

def vqa_accuracy(preds, gts):
    correct = 0
    total = 0
    for pred, gt in zip(preds.values(), gts.values()):
        if pred == gt:
            correct += 1
        total += 1
    return correct / total

def scst_train_step(model, data, optimizer, cider, tokenizer, vocab):
    model.train()
    detections, questions, answers = data
    detections, questions, answers = detections.to(device), questions.to(device), answers.to(device)

    # Generate baseline sequence (greedy decoding)
    with torch.no_grad():
        baseline_out, _ = model.beam_search(detections, questions, 20, vocab.stoi['<eos>'], 5, out_size=1)
        baseline_out = baseline_out[:, 0]

    # Generate sampled sequence (sampling decoding)
    sampled_out, log_probs = model.sample(detections, questions, 20, vocab.stoi['<eos>'], 5)
    
    # Compute rewards
    baseline_captions = tokenizer.decode(baseline_out, skip_special_tokens=True)
    sampled_captions = tokenizer.decode(sampled_out, skip_special_tokens=True)
    rewards = cider.compute_score({0: baseline_captions}, {0: sampled_captions})

    # Compute SCST loss
    scst_loss = -torch.mean((rewards - torch.mean(rewards)) * log_probs)

    # Backpropagation
    optimizer.zero_grad()
    scst_loss.backward()
    optimizer.step()

    return scst_loss.item()

def save_checkpoint(model, optimizer, epoch, loss, accuracy, path='checkpoint.pth'):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    torch.save(state, path)
    print(f'Model saved to {path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_last', action='store_true')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data fields
    image_field = ImageDetectionsField(detections_path='path/to/detections')
    question_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy')
    answer_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy')

    # Dataset and Dataloader
    dataset = OpenViVQA(image_field, question_field, answer_field, 'path/to/annotations', 'path/to/images')
    dataloader_train, dataloader_val = DataLoader(dataset, batch_size=32, shuffle=True), DataLoader(dataset, batch_size=32)

    # Model
    encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory, attention_module_kwargs={'m': 40})
    decoder = MeshedDecoder(len(answer_field.vocab), 54, 3)
    model = Transformer(image_field.vocab_size, encoder, decoder).to(device)

    # Loss and Optimizer
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
