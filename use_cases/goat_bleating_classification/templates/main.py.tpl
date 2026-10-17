"""
main.py
Training VGGish model
UC - GIM - Version 1.0
Author: Marina Cordovilla
"""

import train_model as train
import FeatureParams as params
import os

def main():
    param=params.FeatureParams()

    print(param)
    train.train_model(params=param, dataset_path=@DATASET_PATH@,model_save_path=@RESULT_PATH@, epochs=@EPOCHS@, batch_size=@BATCH_SIZE@, report=True)

if __name__ == "__main__":
    main()
    


