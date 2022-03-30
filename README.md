# TrafficLight_Recognization_Temporal
- Traffic Light Recognization with Temporal data.

# Dataset of Temporal data (20220330)
- You can download dataset from DRIVE: (https://drive.google.com/file/d/1ZQe18rX6ilrpxEruoFVR7HvWfGd6CFba/view?usp=sharing)
- This data includes 7 different classes of Traffic light.
- Currently we are working to extend dataset classes and dataset size.

# Training model
- This code supports the:
>> - Simple Conv with LSTM 
>> - Resnet18 with LSTM
>> - TSN model (https://arxiv.org/pdf/1705.02953.pdf) 
>> - TSM model (https://tinyml.mit.edu/projects/tsm/)
>>> - TSN and TSM source code from (https://github.com/mit-han-lab/temporal-shift-module).
# Training method
- training:
> python3 trainTL_TEMPORAL.py --train_path {path of train dataset} --valid_path {path of valid dataset} --model_type resnetLSTM
--epochs 50 --batch_size 32 --num_frames 10

# Evaluation method
- You can evaluate this method with Detection model.
- Inference code is in Github: (https://github.com/KETIAUTO/TrafficLight_Detection_Inference.git)
