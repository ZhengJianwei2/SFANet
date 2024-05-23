# SFANet
This project provides the code and results for "ORSI Salient Object Detection via Progressive Semantic Flow and Uncertainty-aware Refinement", IEEE TGRS, vol. 62, pp. 5608013–5608025, 2024. If you have any questions, please feel free to contact us："qyq@zjut.edu.cn" or "zjw@zjut.edu.cn"
## Network Architecture 
In this paper, we propose SFANet, a new model for salient object detection in remote sensing images. SFANet mainly comprises four new modules: Semantic Extraction Module (SEM), Interscale Fusion Module (IFM), Deep Semantic Graph-Inference Module (DSGM), and Uncertainty-aware Refinement Module (URM), which address various challenges in RSI-SOD. Specifically, SEM extracts semantic flow knowledge from low-level features, IFM refines and fuses low-level semantic flow information, and DSGM exploits deep semantics to adapt to the complexity of the detection object sizes and shapes. URM introduces edge-guided feature learning to obtain richer textures and reduce the interference of irrelevant objects. Extensive experiments on three RSI-SOD datasets validate the superiority of our proposal. 

<p float="left">
  <img src="/img/SFANet.png" width="800" />
</p>

## Saliency maps
We provide [saliency maps](https://pan.baidu.com/s/1OspaxsovAgyFyin0hLpO-A) (code: qyqq) on ORSSD, EORSSD, and ORSI4199 datasets. In addition, we also provide [measure results (.mat)](https://pan.baidu.com/s/1Mo5xzyAN7gx8VBjliVsWbg) (code: qyqq) on the three datasets.
<p float="left">
  <img src="/img/result1.png" width="800" />
</p> 

## Evaluation Tool
You can use the [evaluation tool (MATLAB version)](https://github.com/MathLee/MatlabEvaluationTools) to evaluate the above saliency maps.

<!-- <p float="left">
  <img src="/img/result1.png" width="800" />
  <img src="/img/result2.png" width="800" />
</p> -->

<!-- ## Viusal results on WDC dataset with 90% missing
![image](https://github.com/ZhengJianwei2/SFANet/img/result2.png)
<!-- ## The spectral and spatial consistency on WDC data under 90% missing rate.
![image](https://github.com/ZhengJianwei2/WHGL/blob/main/img/10SMF_125-28WHGL_b40.png) -->
