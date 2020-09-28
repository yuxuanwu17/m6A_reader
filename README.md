# Prediction of m6A Reader substrate sites using deep convolutional and recurrent neural network  
## Abstract
N6-methyladenosine (m6A) is the most prevalent post-transcriptional modification in mRNA since it could regulate some significant biological functions with the binding of some m6A reader proteins. Multiple readers exist in the human genome, however, the binding specificity was not clarified due to the limited wet experiments on this topic. Therefore, we devised a deep learning approach which incorporated CNN and RNN frameworks together to predict the epitranscriptome-wide targets of six m6A reader proteins (YTHDF1-3, YTHDC1-2, EIF3A). We also utilized layer-wise relevance calculation to obtain contribution of each input feature. Our model achieved state-of-the-art performance with the average AUROC of 0.942 in EIF3A full transcript, compared with 0.929 in CNN-only framework and 0.817 in Support Vector Machine (SVM) method under same condition. Besides, we identified the optimal sequence length (1001bp) in the m6A reader substrate prediction. The results provide new insight into epitranscriptome target prediction and functional characterization of m6A readers.

## Prerequisite packages
keras version 2.3.0, numpy, pandas, argparse

## Run 'python3 main.py -h' in command line for help

### Select the input genes
You could select 6 genes, including YTHDC1-2, YTHDF1-3, EIF3A

### Input for the sequence
You could choose 251, 501, 1001, 2001bp input length to compare the prediction performance 

### The condition of the seqeunce full transcript (full) or mature RNA (exon)
To minimize the bias in selecting the polyA RNAs, we prepared the full transcript data and mature RNA data. In detail, mature RNA data exclude the sites on the intron region while the full transcript data covered either the exon or intron region.

### Choose the CNN or CNN+RNN model
To compare the performance under different framework
