# GLADC
This is a modified version of the code from the paper "Deep Graph Level Anomaly Detection with Contrastive Learning".
# Data Preparation
JOURNAL is the dataset used in the report.
# Train 
     !python3 main_1.py --datadir "./dataset/" --DS "JOURNAL" --hidden-dim 256 --output-dim 128 --num-gc-layers 2 --num_epochs 100 --batch-size 300 --lr 0.0001
#### Credits go to the original authors for the starting code.
        @article{luo2022deep,
        title={Deep graph level anomaly detection with contrastive learning},
        author={Luo, Xuexiong and Wu, Jia and Yang, Jian and Xue, Shan and Peng, Hao and Zhou, Chuan and Chen, Hongyang and Li, Zhao and Sheng, Quan Z},
        journal={Scientific Reports},
        volume={12},
        number={1},
        pages={19867},
        year={2022},
        publisher={Nature Publishing Group UK London}
        }
