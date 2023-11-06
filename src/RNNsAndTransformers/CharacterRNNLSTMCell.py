import time
import random
import unidecode
import string
import re
import os

import matplotlib.pyplot as plt
import torch

torch.backends.cudnn.deterministic = True

# Settings
RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device('cpu')
TEXT_PORTION_SIZE = 200
NUM_ITER = 5000
LEARNING_RATE = 0.005
EMBEDDING_DIM = 100
HIDDEN_DIM = 100

print('Device:', DEVICE)
print("Current Working Directory:", os.getcwd())

# Dataset
print(string.printable)

with open('E:\\dev\\SebRaschka-Lectures\\src\\RNNsAndTransformers\\covid19faq.txt', 'r', errors='ignore') as f:
    textfile = f.read()

# Convert special characters to ASCII
textfile = unidecode.unidecode(textfile)

# Strip extra whitespaces
textfile = re.sub(' +', ' ', textfile)

TEXT_LENGTH = len(textfile)

print(f'Number of characters in text: {TEXT_LENGTH}')

random.seed(RANDOM_SEED)


# Divide the text into smaller portions:
def random_portion(textfile):
    start_index = random.randint(0, TEXT_LENGTH - TEXT_PORTION_SIZE)
    end_index = start_index + TEXT_PORTION_SIZE + 1
    return textfile[start_index:end_index]


print(random_portion(textfile))


def char_to_tensor(text):
    lst = [string.printable.index(c) for c in text]
    tensor = torch.tensor(lst).long()
    return tensor

print(char_to_tensor('abcDEF'))

