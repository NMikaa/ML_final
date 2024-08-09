from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from scipy.sparse import lil_matrix
import torch.optim as optim

from glove.config import Config

class GloVe(nn.Module):
    """
    GloVe მოდელის კლასი.

    ეს კლასი ახორციელებს GloVe (Global Vectors for Word Representation) მოდელს,
    რომელიც გამოიყენება სიტყვების ვექტორული რეპრეზენტაციების შესაქმნელად.
    """

    def __init__(self, vocab_size, embedding_dim):
        """
        ინიციალიზაცია GloVe მოდელისთვის.

        პარამეტრები:
        vocab_size (int): ლექსიკონის ზომა.
        embedding_dim (int): ემბედინგის განზომილება.
        """
        super(GloVe, self).__init__()
        self.wi = nn.Embedding(vocab_size, embedding_dim)
        self.wj = nn.Embedding(vocab_size, embedding_dim)
        self.bi = nn.Embedding(vocab_size, 1)
        self.bj = nn.Embedding(vocab_size, 1)

        # ინიციალიზაცია წონებისთვის
        nn.init.xavier_uniform_(self.wi.weight)
        nn.init.xavier_uniform_(self.wj.weight)
        nn.init.zeros_(self.bi.weight)
        nn.init.zeros_(self.bj.weight)

    def forward(self, i_idx, j_idx):
        """
        წინ გავლა GloVe მოდელში.

        პარამეტრები:
        i_idx (Tensor): სამიზნე სიტყვების ინდექსები.
        j_idx (Tensor): კონტექსტის სიტყვების ინდექსები.

        დაბრუნება:
        Tensor: პროგნოზირებული ლოგ-თანაშემოხვედრის მნიშვნელობები.
        """
        wi = self.wi(i_idx)
        wj = self.wj(j_idx)
        bi = self.bi(i_idx).squeeze()
        bj = self.bj(j_idx).squeeze()
        return (wi * wj).sum(1) + bi + bj


def glove_loss(predictions, targets, x_max, alpha):
    """
    GloVe დანაკარგის ფუნქცია.

    პარამეტრები:
    predictions (Tensor): პროგნოზირებული ლოგ-თანაშემოხვედრის მნიშვნელობები.
    targets (Tensor): ფაქტობრივი თანაშემოხვედრის რაოდენობა.
    x_max (float): მაქსიმალური მნიშვნელობა წონის ფუნქციისთვის.
    alpha (float): ექსპონენტა წონის ფუნქციისთვის.

    დაბრუნება:
    Tensor: გამოთვლილი დანაკარგი.
    """
    weight = torch.where(targets < x_max, (targets / x_max) ** alpha, torch.ones_like(targets))
    loss = weight * (predictions - torch.log1p(targets)) ** 2
    return loss.mean()


def build_cooccurrence_matrices(tokenized_texts, window_size=5, train_ratio=0.8):
    """
    თანაშემოხვედრის მატრიცების შექმნა მოცემული ტოკენიზებული ტექსტებისთვის.

    პარამეტრები:
    tokenized_texts (list of list of str): ტოკენიზებული ტექსტური დოკუმენტების სია.
    window_size (int): კონტექსტის ფანჯრის ზომა.
    train_ratio (float): სასწავლო ნაწილის პროპორცია.

    დაბრუნება:
    tuple: (სასწავლო მატრიცა, ვალიდაციის მატრიცა, სიტყვა-ინდექსის ლექსიკონი)
    """
    vocab = list(set([item for sublist in tokenized_texts for item in sublist]))
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    vocab_size = len(word_to_idx)

    train_matrix = lil_matrix((vocab_size, vocab_size), dtype=np.float32)
    valid_matrix = lil_matrix((vocab_size, vocab_size), dtype=np.float32)

    train_size = int(len(tokenized_texts) * train_ratio)

    for i, tokenized_text in enumerate(tqdm(tokenized_texts)):
        current_matrix = train_matrix if i < train_size else valid_matrix
        for i in range(len(tokenized_text)):
            for j in range(max(0, i - window_size), min(i + window_size + 1, len(tokenized_text))):
                if i != j:
                    current_matrix[word_to_idx[tokenized_text[i]], word_to_idx[tokenized_text[j]]] += 1

    return train_matrix, valid_matrix, word_to_idx


def tokenize_text(corpus, tokenizer):
    """
    ტექსტის ტოკენიზაცია ტოკენიზატორის გამოყენებით.

    პარამეტრები:
    corpus (list of str): ტექსტური დოკუმენტების სია.
    tokenizer: ტოკენიზაციისთვის გამოსაყენებელი ტოკენიზატორი.

    დაბრუნება:
    list of list of str: ტოკენიზებული ტექსტები.
    """
    tokenized_texts = []
    for text in tqdm(corpus):
        tokens = tokenizer.tokenize(text)
        tokenized_texts.append(tokens)
    return tokenized_texts


def train_glove(train_matrix, valid_matrix, config: Config):
    """
    GloVe მოდელის სწავლება.

    პარამეტრები:
    train_matrix: სასწავლო თანაშემოხვედრის მატრიცა.
    valid_matrix: ვალიდაციის თანაშემოხვედრის მატრიცა.
    config (Config): კონფიგურაციის ობიექტი სწავლების პარამეტრებით.

    დაბრუნება:
    GloVe: ნასწავლი GloVe მოდელი.
    """
    vocab_size = train_matrix.shape[0]
    model = GloVe(vocab_size, config.embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # თანაშემოხვედრის მატრიცების გადაყვანა ტენზორებში
    train_i_idx, train_j_idx = np.nonzero(train_matrix)
    train_counts = train_matrix[train_i_idx, train_j_idx]
    train_targets = torch.tensor(train_counts, dtype=torch.float32)
    train_i_idx = torch.tensor(train_i_idx, dtype=torch.long)
    train_j_idx = torch.tensor(train_j_idx, dtype=torch.long)

    valid_i_idx, valid_j_idx = np.nonzero(valid_matrix)
    valid_counts = valid_matrix[valid_i_idx, valid_j_idx]
    valid_targets = torch.tensor(valid_counts, dtype=torch.float32)
    valid_i_idx = torch.tensor(valid_i_idx, dtype=torch.long)
    valid_j_idx = torch.tensor(valid_j_idx, dtype=torch.long)

    best_valid_loss = float('inf')
    no_improvement = 0

    for epoch in range(config.epochs):
        # სწავლების რეჟიმი
        model.train()
        optimizer.zero_grad()
        train_predictions = model(train_i_idx, train_j_idx)
        train_loss = glove_loss(train_predictions, train_targets, config.x_max, config.alpha)
        train_loss.backward()
        optimizer.step()

        # შეფასების რეჟიმი
        model.eval()
        with torch.no_grad():
            valid_predictions = model(valid_i_idx, valid_j_idx)
            valid_loss = glove_loss(valid_predictions, valid_targets, config.x_max, config.alpha)

        # ადრეული შეჩერების ლოგიკა
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= config.patience:
            print(f'ადრეული შეჩერება ეპოქაზე {epoch}')
            break

        if epoch % config.verbose == 0:
            print(f'ეპოქა: {epoch}, სასწავლო დანაკარგი: {train_loss.item():.4f}, ვალიდაციის დანაკარგი: {valid_loss.item():.4f}')

    return model
