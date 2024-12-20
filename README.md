# Depth-Estimation-Using-Intel-RealSense-LiDAR-Camera-L515

This project aims to enhance depth estimation for real-time applications by integrating RealSense
and MiDaS depth models. Combining depth information from an Intel RealSense camera and
MiDaS, a state-of-the-art deep learning model for monocular depth estimation, allows for a
comparative analysis of their depth predictions. Key techniques used include error metrics such as
RMSE and MAE, along with an Exponential Moving Average (EMA) filter to smooth depth
measurements.

#Introduction
Depth estimation is crucial in fields such as robotics, autonomous navigation, and augmented reality.
While sensor-based depth estimation provides direct measurements, deep learning models like
MiDaS estimate depth from RGB images alone. This project assesses the efficacy of both methods
using real-time data, comparing their depth predictions and applying error metrics to gauge accuracy.

#Objectives
1. RealSense Depth Integration: Use an Intel RealSense camera to capture real-time depth
information.
2. Deep Learning Model Application: Integrate MiDaS for monocular depth estimation.
3. Error Measurement: Calculate RMSE and MAE between RealSense and MiDaS
predictions.
4. Filter Application: Implement EMA filtering for smoothed distance calculations.

#Dataset
Data is collected from an Intel RealSense camera in real-time, configured for both depth and color
streaming. RGB frames are processed through MiDaS, while RealSense provides depth frames. Both
streams are preprocessed for alignment and evaluation.

#Methodology
1. Pipeline Initialization: A RealSense pipeline is established, and color and depth streams are
activated at 640x480 and 320x240 resolutions, respectively.
2. MiDaS Model Loading: MiDaS is loaded using Torch Hub, with a DPT_Large architecture
providing high-quality depth predictions.
3. Transformations and Normalization: Frames are transformed and normalized for
consistency before feeding into MiDaS.
4. Error Metrics: RMSE and MAE metrics are used to assess accuracy between RealSense depth
maps and MiDaS predictions.Data Collection and Preprocessing.

#Model Compilation and Training
MiDaS operates in inference mode, predicting depth on a frame-by-frame basis, while RealSense
depth frames are directly retrieved. The EMA filter, implemented to reduce frame-to-frame depth
variance, allows for a smoother real-time depth estimation.

#Results
Error Metrics: RMSE and MAE calculations reveal the depth prediction accuracy of MiDaS
relative to RealSense. Sample results include:
• RMSE
• MAE

#Filtered Depth Results: EMA-filtered values are presented for both MiDaS and RealSense,
demonstrating smoother transitions and fewer fluctuations.
Visualization: Depth maps from MiDaS and RealSense are color-mapped and displayed side-by-side
to illustrate differences.
