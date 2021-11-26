## DeepAoANet

This project aims to providing **a data-driven method to estimating multiple AoAs from single snapshot of channel covariance matrix** using the KerberosSDR, which comprises four RTL-SDRs. The proposed DeepAoANet is found over 99% accuracy of classifying number of impinging signals, and sub-degree level of RMSE or MAE. The training split of our dataset can be found [here](https://drive.google.com/drive/folders/1421NOSQcveTE-TpKAM6cPg3SN_vatJPQ?usp=sharing).

Benchmark performance of DeepAoANet-FC, DeepAoANet-CNN, MUSIC, and Support Vector Regression are shown below.

![cdf1606](https://github.com/zdai257/GPSLoRaRX/blob/main/doc/CDF_Xx-ym4.png)

Error distributions using DeepAoANet-FC and DeepAoANet-CNN are as below. It can seen DeepAoANets have stability across a wide Field of View.

![scatter](https://github.com/zdai257/GPSLoRaRX/blob/main/doc/Shaded_deepaoanet.png)

Both DeepAoANet-FC and DeepAoANet-CNN show resilience with negative Signal-to-Noise Ratios lower than those of the source dataset.

![scatter](https://github.com/zdai257/GPSLoRaRX/blob/main/doc/SNR4.png)
