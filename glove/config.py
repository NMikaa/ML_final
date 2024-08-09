from dataclasses import dataclass

@dataclass
class Config:
    embedding_dim = 50
    window_size = 5
    x_max = 100
    alpha = 0.75
    epochs = 100
    learning_rate = 0.05
    verbose = 1
    patience = 5
