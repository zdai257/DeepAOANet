## DeepAoANet

This project aims to providing *a data-driven method to estimating multiple AoAs from single snapshot of channel covariance matrix* using the **KerberosSDR**, which comprises four RTL-SDRs. The proposed DeepAoANet is found over 99% accuracy of classifying number of impinging signals (0, 1, or 2), and sub-degree level of regression error.

Our collected dataset can be found [here](https://aston.box.com/s/yvmfecgln5nkgxde7bmytgircwsnm2nf). Any question please contact Z. Dai (z.dai1@aston.ac.uk).

Please cite our journal paper as

```
@ARTICLE{dai_deepaoanet2022,
  author={Dai, Zhuangzhuang and He, Yuhang and Tran, Vu and Trigoni, Niki and Markham, Andrew},
  journal={IEEE Access}, 
  title={DeepAoANet: Learning Angle of Arrival From Software Defined Radios With Deep Neural Networks}, 
  year={2022},
  volume={10},
  number={},
  pages={3164-3176},
  doi={10.1109/ACCESS.2021.3140146}}
```

### Usage

Prerequisites include

* rospy
* rosbag
* keras 2.2.4
* PyQt5
* pyqtgraph
* sklearn

Download the DeepAoANet dataset. Use the following to create augmented rosbags of AoA(s) from raw AoA rosbags

```bash
python Create_Synthetic.py
python Multilabel_Create_Synthetic.py
```

Use the following to convert rosbags to dataframes, saved as .pkl

```bash
python AOAsingle_main.py
python AOAmultilabel_main.py
```

Load all .pkl, use the following for training

```bash
python AOAtrain.py
```

Connect and configure **KerberosSDR**, run the any of the following GUIs for AoA visualization. Note raw data was collected with a LoRa transmitter. Other signals may not generalize well. Finetuning with local signals is recommended.

```bash
python DeepAOAIE.py
# OR
python DeepAOAPolar.py
# OR
python DeepAOAclassifier.py
```

### Performance

Benchmark performance of DeepAoANet-FC, DeepAoANet-CNN, MUSIC, and Support Vector Regression are shown below.

![cdf1606](https://github.com/zdai257/GPSLoRaRX/blob/main/doc/CDF_Xx-ym4.png)

Error distributions using DeepAoANet-FC and DeepAoANet-CNN are as below. It can seen DeepAoANets have stability across a wide Field of View.

![scatter](https://github.com/zdai257/GPSLoRaRX/blob/main/doc/Shaded_deepaoanet.png)

Both DeepAoANet-FC and DeepAoANet-CNN show resilience with negative Signal-to-Noise Ratios lower than those of the source dataset.

![scatter](https://github.com/zdai257/GPSLoRaRX/blob/main/doc/SNR4.png)
