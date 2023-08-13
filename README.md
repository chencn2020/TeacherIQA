# TeacherIQA

![visitors](https://visitor-badge.laobi.icu/badge?page_id=chencn2020/TeacherIQA)
[![paper](https://img.shields.io/badge/arXiv-Paper-green.svg)](https://openaccess.thecvf.com/content/ACCV2022/html/Chen_Teacher-Guided_Learning_for_Blind_Image_Quality_Assessment_ACCV_2022_paper.html)
[![download](https://img.shields.io/github/downloads/chencn2020/TeacherIQA/total.svg)](https://github.com/chencn2020/TeacherIQA/releases)
[![Open issue](https://img.shields.io/github/issues/chencn2020/TeacherIQA)](https://github.com/chencn2020/TeacherIQA/issues)
[![Closed issue](https://img.shields.io/github/issues-closed/chencn2020/TeacherIQA)](https://github.com/chencn2020/TeacherIQA/issues)
[![GitHub Stars](https://img.shields.io/github/stars/chencn2020/TeacherIQA?style=social)](https://github.com/chencn2020/TeacherIQA)

This repository is the source code for the paper "[Teacher-Guided Learning for Blind Image Quality Assessment](https://openaccess.thecvf.com/content/ACCV2022/html/Chen_Teacher-Guided_Learning_for_Blind_Image_Quality_Assessment_ACCV_2022_paper.html)".

![Framework](./pic/framework.jpg)

## Dependencies

- matplotlib==3.2.2
- numpy==1.22.3
- Pillow==9.2.0
- torch==1.11.0
- torchvision==0.11.2+cu113

## Usages For Testing


You can predict the image quality score for any images with our model which is trained on KonIq-10k dataset.

The pre-trained model can be downloaded from [Google drive](https://drive.google.com/file/d/1iNhJQpUWSAkwSfDbfXzu834gm7NoT3m0/view?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1aE8_stfHexjzPECyk1YlwA) (password: b86d).

Please put the pre-trained model into **'./checkpoint'** folder. Then run:

```
python3 demo.py --input_image ./demoImg/I_02.jpg --pre_train_model ./checkpoint/koniq_teacher_iqa.pkl --crop_times 25
```

The input image will be randomly crop into 25 patches in size 224 × 224 and the IQA model will predict 25 scores for each patches.

Finally, you will get an average quality score ranging from 0-100. But there exists some cases whose the value may be out of the range. The higher value is, the better image quality is.


## Citation
If our work is useful to your research, we will be grateful for you to cite our paper:
```
@InProceedings{Chen_2022_ACCV,
    author    = {Chen, Zewen and Wang, Juan and Li, Bing and Yuan, Chunfeng and Xiong, Weihua and Cheng, Rui and Hu, Weiming},
    title     = {Teacher-Guided Learning for Blind Image Quality Assessment},
    booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
    month     = {December},
    year      = {2022},
    pages     = {2457-2474}
}

