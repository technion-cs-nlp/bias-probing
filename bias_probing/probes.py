import logging
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import SequentialSampler, DataLoader, Dataset
from tqdm import tqdm, trange

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)


def save_checkpoint(output_dir, train_batch_size, num_train_epochs, model, optimizer, loss, epoch, tag='checkpoint'):
    path = os.path.join(output_dir,
                        '{}_b{}_e{}.pt'.format(tag, train_batch_size, num_train_epochs))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def load_checkpoint(output_dir, train_batch_size, num_train_epochs, model, optimizer, tag='checkpoint') -> Tuple[
    nn.Module, optim.Optimizer, int, float]:
    """
    Load a model from checkpoint for a given output_dir with given arguments.
    model_args must contain train_batch_size and num_train_epochs, and they uniquely identify the model checkpoint.

    Returns: a tuple (model, optimizer, epoch, loss) where model and optimizer are preloaded
    """
    if output_dir is None:
        return model, optimizer, 0, 0.0
    path = os.path.join(output_dir,
                        '{}_b{}_e{}.pt'.format(tag, train_batch_size, num_train_epochs))
    if os.path.exists(path):
        logger.info('Found checkpoint %s, loading model and optimizer...', path)
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        logger.info('Last checkpoint: loss=%f, epoch=%d', loss, epoch)
        return model, optimizer, epoch, loss
    else:
        logger.info('No checkpoint %s, creating fresh model...', path)
        return model, optimizer, 0, 0.0


@dataclass
class ProbeTrainingArgs:
    train_batch_size: int
    learning_rate: float
    num_train_epochs: int
    checkpoint_steps: int
    early_stopping: int
    early_stopping_tolerance: float


def train_probe(args: ProbeTrainingArgs, train_dataset: Dataset, model: nn.Module,
                dev_dataset=None, loss_fn=nn.CrossEntropyLoss(),
                checkpoint_path=None, collate_fn=None, device=None, verbose=False):

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=collate_fn)

    # Training parameters
    probe_classifier = model.to(device)
    criterion = loss_fn.to(device)
    optimizer = optim.AdamW(probe_classifier.parameters(), lr=args.learning_rate)

    loss_list = []
    acc_list = []
    tr_loss = 0.0

    probe_classifier, optimizer, epoch, checkpoint_loss = load_checkpoint(checkpoint_path, args.train_batch_size,
                                                                          args.num_train_epochs, probe_classifier,
                                                                          optimizer)

    if checkpoint_path is not None and not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    probe_classifier = probe_classifier.to(device)
    train_iterator = trange(epoch, int(args.num_train_epochs), disable=not verbose)
    epochs_without_improvement = 0
    min_dev_loss = np.inf
    mean_loss = 0

    best_model = None
    for _ in train_iterator:
        acc = 0
        total_size = 0
        mean_loss = 0

        num_batches = 0
        epoch_iterator = tqdm(train_dataloader, desc='Iteration', disable=True)
        probe_classifier.train()
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(Variable(t).to(device) for t in batch)
            # Load batch
            ids = batch[0]
            embedding = batch[1]
            labels = batch[2]

            # Forward propagation
            optimizer.zero_grad()
            outputs = probe_classifier(embedding)

            # Backward propagation
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate loss and accuracy
            l_item = loss.item()
            _, prediction = torch.max(outputs, 1)
            correct = (prediction == labels).sum().item()
            acc += correct
            tr_loss += l_item
            mean_loss += l_item
            total_size += embedding.size()[0]
            num_batches += 1
            epoch_iterator.set_description('Loss: {}'.format(l_item))

        mean_loss /= num_batches
        epoch += 1
        if epoch % args.checkpoint_steps == 0 and checkpoint_path is not None:
            # Save a checkpoint
            save_checkpoint(checkpoint_path, args.train_batch_size, args.num_train_epochs, probe_classifier, optimizer, mean_loss,
                            epoch)

        acc = acc / total_size
        acc_list.append(acc)
        loss_list.append(mean_loss)

        if dev_dataset is not None:
            # Validation
            probe_classifier.eval()
            dev_loss, _, _ = evaluate_probe(args, dev_dataset, probe_classifier,
                                            loss_fn=loss_fn,
                                            device=device)
            train_iterator.set_description(
                'Accuracy: {:.2f}%, Loss: {:.3f}, Validation Loss: {:.3f}'.format(acc * 100, mean_loss, dev_loss))
            if dev_loss <= min_dev_loss - args.early_stopping_tolerance:
                epochs_without_improvement = 0
                min_dev_loss = dev_loss
                best_model = probe_classifier.state_dict()
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement > args.early_stopping:
                    logger.info(f'Early stopping after {epoch} epochs since no validation improvement achieved')
                    break
        else:
            train_iterator.set_description('Accuracy: {:.2f}%, Loss: {:.4f}'.format(acc * 100, mean_loss))

    logger.info(f'Stopped after {epoch}')
    # Load best model
    model.load_state_dict(best_model)
    if checkpoint_path is not None:
        save_checkpoint(checkpoint_path, args.train_batch_size, args.num_train_epochs, probe_classifier, optimizer, mean_loss,
                        epoch, tag='final')
    return acc_list, loss_list


def evaluate_probe(args: ProbeTrainingArgs,
                   eval_dataset: Dataset,
                   model,
                   loss_fn=None,
                   verbose=False,
                   collate_fn=None,
                   device=None):
    """Evaluate a probe model on a given dataset.

    :param args: Arguments for probe training
    :param eval_dataset: The dataset to evaluate on
    :param model: The (trained) probe model
    :param loss_fn: The loss function, Default: CrossEntropyLoss
    :param verbose: If true, prints progress bars and logs. Default: False
    :param collate_fn: A collate function passed to PyTorch `DataLoader`
    :param device:
    :return: A tuple (loss, out_label_ids, preds)
    * loss: mean loss over all samples
    * out_label_ids: The dataset labels, of shape (N,)
    * preds: The model predictions, of shape (N, C), where C is the number of classes that the model predicts
    """
    if eval_dataset is None:
        raise ValueError('eval_dataset cannot be None')
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss(reduction='sum')

    tr_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    criterion = loss_fn
    test_sampler = SequentialSampler(eval_dataset)
    test_dataloader = DataLoader(eval_dataset, batch_size=args.train_batch_size,
                                 sampler=test_sampler, collate_fn=collate_fn)

    for batch in tqdm(test_dataloader, desc='Evaluating', disable=not verbose):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            ids = batch[0]
            embedding = batch[1]
            labels = batch[2]

            outputs = model(embedding)

        loss = criterion(outputs, labels)
        tr_loss += loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = outputs.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, outputs.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

    return tr_loss, out_label_ids, preds
