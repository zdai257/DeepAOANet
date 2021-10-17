## GPS-LoRa RX postprocessing and A data-driven AOA method

This repo aims to providing a **data-driven method to estimating multiple AOAs from single snapshot of channel covariance matrix** using the KerberosSDR, which comprises four RTL-SDRs. DeepAOANet is found to classify number of impinging signals with over $99\%$ accuracy and sub-degree level of RMSE or MAE.

Benchmark performance of DeepAOANet-FC, DeepAOANet-CNN, MUSIC, and Support Vector Regression are shown below.

![cdf1606](https://github.com/zdai257/GPSLoRaRX/blob/main/doc/CDF_Xx-ym3.png)

Error distributions using DeepAOANet-FC and DeepAOANet-CNN are as below. It can seen DeepAOANets have a wide and resilient Field of View.

![scatter](https://github.com/zdai257/GPSLoRaRX/blob/main/doc/Scatter_test.png)
