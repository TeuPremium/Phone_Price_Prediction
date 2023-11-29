# Phone Price Prediction Neural Network

## Project Overview

To test and run this project, it is necessary to execute the run_network_3HL.py archive.
Check out the presentations in [PDF](https://github.com/TeuPremium/Phone_Price_Prediction/blob/main/Presentation.pdf) or [PPTX](https://github.com/TeuPremium/Phone_Price_Prediction/blob/main/Projec%20Presentation.pptx)!

### Dataset Source
This project utilizes the dataset from [Kaggle](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification), specifically focusing on the problem of mobile price classification.

## About the Dataset
### Context
Bob has started his own mobile company, aiming to compete with big players like Apple and Samsung. However, he struggles to estimate the prices of the mobiles his company creates. To address this, he collects sales data from various companies and seeks to establish a relationship between mobile features (e.g., RAM, Internal Memory) and selling prices.

## Project Objectives
The objective of this project was classification on the labeled training dataset, and analyzing the quality of the results. For this reason, the network's performance was exclusively verified on the train dataset. However, it's worth noting that this neural network can also predict for the unlabeled test dataset if applied to such a task.

## Model Performance
### Accuracy Results:
- Train Accuracy: 100.00%
- Validation Accuracy: 89.50%
- Test Accuracy: 94.00%

To view results by the network in more detail, check out the [presentation](https://github.com/TeuPremium/Phone_Price_Prediction/blob/main/Presentation.pdf).

## Training Dataset RMS Error per Epoch
![Training Dataset RMS Error](https://github.com/TeuPremium/Phone_Price_Prediction/assets/50275359/3435effe-c729-4475-848b-c1981feaca4e)

## Implementation Details
To run this project and verify the results, execute the "Run_network_3HL.py" script, ensuring that Python and the required libraries (numpy, pandas, matplotlib) are installed.

This project was developed using the nnfs book and online course available on YouTube as a source for network construction.

## Links
- Kaggle Dataset: [Mobile Price Classification](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)
- nnfs Book: [nnfs.io](https://nnfs.io/)
