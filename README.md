# CRP_DAM
"Dynamic Attention Model – A Deep Reinforcement Learning Approach for Container Relocation Problem"
This paper has been received by IEAAIE2023. [Here is the paper link.](https://link.springer.com/chapter/10.1007/978-3-031-36822-6_24)

Feel free to raise any issues.

In my actual code implementation, **the higher priority number is bigger**. It is different from the description in paper which follows pervious related papers.  

## Dependencies

- Python >= 3.6
- TensorFlow >= 2.0
- PyTorch = 1.10
- tqdm
- scipy
- numpy

## Install pytorch

If you have not installed pytorch, you can follow my installation code. Or you can install yourself from other source.

```python
pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
```

## Usage

#### Training config

First, generate the pickle file containing hyperparameter values by running the following command.

You can find all hyperparameter meaning in config.py.

Example: 

```shell
python config.py -n 9 -s 3 -t 5 -b 128 -bs -2000 -bv 100 -e 30
```

You will get CRP_9_3_5_train.pkl in /Pkl.

In training data generation，we always generate data like $3*(5-2)=9$ , so there 2 empty tiers in each stack. This setting follows CM dataset. 

https://www.bwl.uni-hamburg.de/en/iwi/forschung/projekte/dataprojekte/brp-instances-caserta-etal-2012.zip

#### Training

```shell
python train.py -p Pkl/CRP_9_3_5_train.pkl
```

You will get CRP_9_3_5_train_epoch29_2.pt in /Weights

"_2.pt" means that we calculate the loss using meaning of result the training batch now. We find the performance better do this rather than use greedy rollout of latest saved baseline. You can easily change the loss calculation by changing Line 56-57 in train.py. 

#### implementation

```shell
python Impr_data_test.py -op data/cm_am_res_24_6_6.txt -tx data/cm_test_am_24_6_6.txt -p Weights/CRP_20_5_6_train_epoch29_2.pt -sm 240 -nm 40 -n 24 -s 6 -t 6
```

-op: result output_path 

-tx:  data

-p:  use which model weight

-sm --sampl_num :  sum of sample num

-nm  --impr_num  :  num of permutation for instance augmentation

-n -s -t : same meaning as training config.

We propose a new augmentation method that shuffles the stacks of an instance. Then test for the same total sample times.

I can't explain why this work because AM model is invariant to the order of
the inputs (i.e., the stacks). (This is the reject reason of ICRA2023).

You can use test.py to not use instance augmentation.

#### Input data production

The detail data format is in function **data_from_txt(path)** in data.py

You can transfer the dataset like (CM dataset) to the proper data format yourself. Don't forget  **the higher priority number is bigger**. So you should reverse the priority number.

Or you can use /data/make_data.cpp to generate random data.

## Reference
- https://github.com/wouterkool/attention-learn-to-route
- https://github.com/Rintarooo/VRP_DRL_MHA
