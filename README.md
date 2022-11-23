
## Evaluating diagnostic content of AI-generated chest radiography: a multi-center visual Turing test

<p align="left"><img width="95%" src="assets/teaser.jpg" /></p>

> **Evaluating diagnostic content of AI-generated chest radiography: a multi-center visual Turing test**<br>
> [Youho Myong]\¶, [Dan Yoon]\¶, [Byeong Soo Kim]\, [Young Gyun Kim]\, [Yongsik Sim]\, [Suji Lee]\, [Jiyoung Yoon]\, [Minwoo Cho]\*, and [Sungwan Kim]*<br>
> PLoS ONE. (¶ indicates equal contribution)<br>

> Paper: ""

> **Abstract:** *[Background]
Accurate interpretation of chest radiographs requires years of medical training, and many countries face a shortage of medical professionals to meet such requirements. Recent advancements in artificial intelligence (AI) have aided diagnoses; however, their performance is often limited due to data imbalance. The aim of this study was to augment imbalanced medical data using generative adversarial networks (GANs) and evaluate the clinical quality of the generated images via a multi-center visual Turing test.
[Methods]
Using six chest radiograph datasets, (MIMIC, CheXPert, CXR8, JSRT, VBD, and OpenI), starGAN v2 generated chest radiographs with specific pathologies. Five board-certified radiologists from three university hospitals, each with at least five years of clinical experience, evaluated the image quality through a visual Turing test. Further evaluations were performed to investigate whether GAN augmentation enhanced the convolutional neural network (CNN) classifier performances.
[Results]
In terms of identifying GAN images as artificial, there was no significant difference in the sensitivity between radiologists and random guessing (result of radiologists: 147/275 (53.5%) vs result of random guessing: 137.5/275, (50%); p=.284). GAN augmentation enhanced CNN classifier performance by 11.7%.
[Conclusion]
Radiologists effectively classified chest pathologies with synthesized radiographs, suggesting that the images contained adequate clinical information. Furthermore, GAN augmentation enhanced CNN performance, providing a bypass to overcome data imbalance in medical AI training. CNN based methods rely on the amount and quality of training data; the present study showed that GAN augmentation could effectively augment training data for medical AI.
*


## TensorFlow implementation
The TensorFlow implementation of StarGAN v2 can be found at [clovaai/stargan-v2-tensorflow](https://github.com/clovaai/stargan-v2-tensorflow).

## Software installation
Clone this repository:

```bash
git clone https://github.com/clovaai/stargan-v2.git
cd stargan-v2/
```

Install the dependencies:
```bash
conda create -n stargan-v2 python=3.6.7
conda activate stargan-v2
conda install -y pytorch=1.4.0 torchvision=0.5.0 cudatoolkit=10.0 -c pytorch
conda install x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge
pip install opencv-python==4.1.2.30 ffmpeg-python==0.2.0 scikit-image==0.16.2
pip install pillow==7.0.0 scipy==1.2.1 tqdm==4.43.0 munch==2.5.0
```


## Training networks
To train StarGAN v2 from scratch, run the following commands. Generated images and network checkpoints will be stored in the `expr/samples` and `expr/checkpoints` directories, respectively. Please see [here](https://github.com/clovaai/stargan-v2/blob/master/main.py#L86-L179) for training arguments and a description of them. 

```bash
# celeba-hq
python main.py --mode train --num_domains 4 --w_hpf 1 \
               --lambda_reg 1 --lambda_sty 1 --lambda_ds 1 --lambda_cyc 1 \
               --train_img_dir /home/ubuntu/dan/stargan-v2/datasets/train \
               --val_img_dir /home/ubuntu/dan/stargan-v2/datasets/val

```



## License
The source code, pre-trained models, and dataset are available under [Creative Commons BY-NC 4.0](https://github.com/clovaai/stargan-v2/blob/master/LICENSE) license by NAVER Corporation. You can **use, copy, tranform and build upon** the material for **non-commercial purposes** as long as you give **appropriate credit** by citing our paper, and indicate if changes were made. 

For business inquiries, please contact clova-jobs@navercorp.com.<br/>	
For technical and other inquires, please contact yunjey.choi@navercorp.com.


## Acknowledgements
Y.M. received grant from by the MD-PhD/Medical Scientist Training Program through the Korea Health Industry Development Institute funded by the Korean government (Ministry of Health and Welfare, http://www.mohw.go.kr/eng/index.jsp). M.C. received grant from National Research Foundation of Korea (NRF) grant funded by the Korean government (Ministry of ICT, Science, and Technology, https://www.msit.go.kr/eng/index.do) (No-2021R1C1C2095529). The funders had no role in study design, data collection and analysis, decision to publish, or preparation of the manuscript.
