# MMHH
The Pytorch implementation of Maximum-Margin Hamming Hashing.

## Requirements

The code requires some common packages:

```shell
# python>=3.6
# Anaconda: it is not necessary but recommended since it contains a lot of packages.
conda create -n py36 python=3.6 
source activate py36

# pytorch = 0.4.1
conda install pytorch=0.4.1 cuda92 -c pytorch

# Maybe you need tensorboardX for detailed analysis
pip install tensorboardX
```

## Data Preparation

We recommend you to follow [HashNet](https://github.com/thuml/HashNet/)  to prepare the dataset images. 

Our paper also conducts experiments on **noise data** and **Unseen Classes Retrieval Protocol**. The preprocessing has been elaborated in the paper and is easy to follow. We will add the preprocessing scripts soon.

## Example Usage

To train the model, it is an example:

```shell
python train_mmhh.py --gpu_id=0 --s_dataset="coco_80" --hash_bit=48 --annotation="MMHH-train" --loss_lambda=0.001 --num_iters=1000 --image_network="AlexNetFc" --batch_size=48 --radius 2 --distance_type "MMHH" --similar_weight "1"  --lr 0.0001 --decay_step 200  --gamma 10.0 --opt-test True
```

The model will be examined in the end of training. If you want to test a model individually, run the following example:

```shell
python test_mmhh.py --gpu_id=0 --dataset="coco_80" --model_path "../snapshot/hash/MMHH-train_coco_80_coco_80_iter_01000" --batch_size=48 --radius 2 --opt-test --annotation="MMHH-test" --test_sample_ratio 1.0
```

The basic metric functions refer to [DeepHash](https://github.com/thulab/DeepHash)  (MAP@H<=2) and [HashNet](https://github.com/thuml/HashNet/) (MAP@TopK). We optimize them carefully, which speed up by $\times 2\sim \times 10$. 

Due to the `numpy` randomness, the optimized version may be slightly different from the original ones, but we believe it doesn't matter after lots of tests.

## Acknowledgments

Our code mainly refer to the following repositories, we want to thanks for their invaluable help sincerely:

*  [HashNet](https://github.com/thuml/HashNet/) : the dataset, data processing, the network backbones, etc..
*  [DeepHash](https://github.com/thulab/DeepHash): the DCH implementation and the training parameters.
* [Snca.pytorch](https://github.com/microsoft/snca.pytorch): the augmented memory.

## Citations

If you find the codes are helpful to your work, please kindly cite our paper:

```
@inproceedings{DBLP:conf/iccv/Kang0L0Y19,
  author    = {Rong Kang and
               Yue Cao and
               Mingsheng Long and
               Jianmin Wang and
               Philip S. Yu},
  title     = {Maximum-Margin Hamming Hashing},
  booktitle = {2019 {IEEE/CVF} International Conference on Computer Vision, {ICCV}
               2019, Seoul, Korea (South), October 27 - November 2, 2019},
  pages     = {8251--8260},
  publisher = {{IEEE}},
  year      = {2019},
}
```

If you encounter any issues, please feel free to send an email to [kangr15@mails.tsinghua.edu.cn](mailto:kangr15@mails.tsinghua.edu.cn). We will do our best to address your concerns.

