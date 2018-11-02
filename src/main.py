# coding=utf-8
import sys
from models.gemini import gemini
DATA_DIR = 'data/dataset/train/gemini/tfmodel/201811021025'
MODEL_DIR = 'data/dataset/model/gemini'
LOG_DIR = 'data/dataset/log/gemini'
def main():
    gemini.train_model(DATA_DIR, MODEL_DIR, LOG_DIR)

if __name__ == '__main__':
    main()