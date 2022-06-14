# Current reposotiry::: TEST VERSION
- includes:
>> Hydra 
>> MlFlor
>> Apex training
>> wandb 
>> various scheduler
>> various optimizer
>> various loss functions.
# TrafficLight_Recognization_Temporal
- Traffic Light Recognization with Single frame data and Temporal frame data.

# Dataset of Temporal data (20220330)
- You can download dataset from Drive [Link](https://drive.google.com/file/d/1ZQe18rX6ilrpxEruoFVR7HvWfGd6CFba/view?usp=sharing)
- This data includes 7 different classes of Traffic light.
- Currently we are working to extend dataset classes and dataset size.

# Training model
- This code supports the:
>> - Simple Conv with LSTM 
>> - Resnet18 with LSTM
>> - TSN model [Paper](https://arxiv.org/pdf/1705.02953.pdf) 
>> - TSM model [Site](https://tinyml.mit.edu/projects/tsm/)
>>> - TSN and TSM source code from [SOURCE](https://github.com/mit-han-lab/temporal-shift-module).
# Training method
- training Temporal Traffic Light Model:   
<pre>
<code>
python trainTL_TEMPORAL.py --train_path {path of train dataset} --valid_path {path of valid dataset} --model_type resnetLSTM
--epochs 50 --batch_size 32 --num_frames 10
</code>

- training Single Frame Traffic Light Model:   
<pre>
<code>
python train.py train_config.model_type=single_frame model.name=resnet18_128x64 train_config.train_bs=256
</code>
</pre>

# Evaluation method
- You can evaluate this method with Detection model.
- Inference code is in [Github](https://github.com/AUTOSELFDRIVING2022/TrafficLight_Detection_Inference.git)
