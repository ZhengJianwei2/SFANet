# SFANet
This project provides the code and results for "ORSI Salient Object Detection via Progressive Semantic Flow and Uncertainty-aware Refinement", IEEE TGRS, vol. 62, pp. 5608013–5608025, 2024. If you have any questions, please feel free to contact us："qyq@zjut.edu.cn" or "zjw@zjut.edu.cn"
## Network Architecture 
* To obtain feature maps with rich semantic information and critical object locations, two individual modules, namely semantic extraction module (SEM) and interscale fusion module (IFM), are meticulously designed that efficiently extract and merge semantic flow from a global
perspective.
* A lightweight interaction-attention module (LIAM) is proposed, which is specifically designed to suppress task-irrelevant information, highlighting the meaningful objects and enriching the representation of details in low-level features of saliency maps. 

<p float="left">
  <img src="/img/SFANet.png" width="800" />
  <img src="/img/result1.png" width="800" />
  <img src="/img/result2.png" width="800" />
</p>

## Saliency maps
We provide saliency maps [bilibili](https://pan.baidu.com/s/1OspaxsovAgyFyin0hLpO-A)(code: qyqq) on ORSSD EORSSD, and ORSI4199 datasets.

<!-- ## Viusal results on WDC dataset with 90% missing
![image](https://github.com/ZhengJianwei2/SFANet/img/result2.png)
<!-- ## The spectral and spatial consistency on WDC data under 90% missing rate.
![image](https://github.com/ZhengJianwei2/WHGL/blob/main/img/10SMF_125-28WHGL_b40.png) -->
