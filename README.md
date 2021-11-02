## DeepAOANet

This project aims to providing **a data-driven method to estimating multiple AOAs from single snapshot of channel covariance matrix** using the KerberosSDR, which comprises four RTL-SDRs. The proposed DeepAOANet is found over 99% accuracy of classifying number of impinging signals, and sub-degree level of RMSE or MAE. The training split of our dataset can be found [here](https://drive.google.com/drive/folders/1421NOSQcveTE-TpKAM6cPg3SN_vatJPQ?usp=sharing).

Benchmark performance of DeepAOANet-FC, DeepAOANet-CNN, MUSIC, and Support Vector Regression are shown below.

![cdf1606](https://github.com/zdai257/GPSLoRaRX/blob/main/doc/CDF_Xx-ym4.png)

Error distributions using DeepAOANet-FC and DeepAOANet-CNN are as below. It can seen DeepAOANets have a wide Field of View.

![scatter](https://github.com/zdai257/GPSLoRaRX/blob/main/doc/Scatter_test.png)

Both DeepAOANet-FC and DeepAOANet-CNN show resilience with negative Signal-to-Noise Ratios.

![scatter](https://github.com/zdai257/GPSLoRaRX/blob/main/doc/SNR4.png)
