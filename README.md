# WEA-DINO

The code of "WEA-DINO: AN IMPROVED DINO WITH WORD EMBEDDING ALIGNMENT FOR REMOTE SENSING SCENE ZERO-SHOT OBJECT DETECTION".

<table>
    <tr>
    <td><img src="PaperFigs\Fig1.png" width = "100%" alt="Cross-Domain RS Semantic Segmentation"/></td>
    <td><img src="PaperFigs\Fig2.png" width = "100%" alt="WEA-DINO"/></td>
    </tr>
</table>

## Dataset Preparation

We select Postsdam, Vaihingen and LoveDA as benchmark datasets. Please put the datasets in ./data folder.

## WEA-DINO

### Install

1. requirements:
    
    python >= 3.7
        
    pytorch >= 1.7
        
    cuda >= 11.0
    
2. prerequisites: Please refer to  [MMDetection PREREQUISITES](https://github.com/open-mmlab/mmdetection).


### Training

./tools/dist_train.sh ./experiments/dino-4scale_r50_8xb2-36e_GZSD_wordvec.py 2


### Testing

Trained with the above commands, you can get a trained model to test the performance of your model.   

./tools/dist_test.sh ./experiments/dino-4scale_r50_8xb2-36e_GZSD.py ./experiments/DIOR_GZSD_results/yourpthmodel.pth 2


If you have any question, please discuss with me by sending email to wanggb@buaa.edu.cn.

# References
Many thanks to their excellent works
* [MMDetection](https://github.com/open-mmlab/mmdetection)
