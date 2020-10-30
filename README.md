# Prediction of m6A Reader substrate sites using deep convolutional and recurrent neural network  
## Abstract
N6-methyladenosine (m6A) is the most prevalent post-transcriptional modification in mRNA since it could regulate some significant biological functions with the binding of some m6A reader proteins. Multiple readers exist in the human genome, however, the binding specificity was not clarified due to the limited wet experiments on this topic. Therefore, we devised a deep learning approach which incorporated CNN and RNN frameworks together to predict the epitranscriptome-wide targets of six m6A reader proteins (YTHDF1-3, YTHDC1-2, EIF3A). We also utilized layer-wise relevance calculation to obtain contribution of each input feature. Our model achieved state-of-the-art performance with the average AUROC of 0.942 in EIF3A full transcript, compared with 0.929 in CNN-only framework and 0.817 in Support Vector Machine (SVM) method under same condition. Besides, we identified the optimal sequence length (1001bp) in the m6A reader substrate prediction. The results provide new insight into epitranscriptome target prediction and functional characterization of m6A readers.

## Material and Methods
### Identification of m6A reader binding sites

For the benefits of modeling, we need to define the positive and negative samples of m6A reader binding site. As can be seen from Figure 1, there are three significant factors to consider when determining the m6A reader binding sites, which is DRACH motif, known m6A Sites and CLIP labeled Sites. Generally, m6A readers have a tendency to bind known m6A sites and DRACH motif, but these two are not robust enough, without CLIP labeled sites, they could still be considered as negative samples. Therefore, in this experiment, three factors have to be satisfied simultaneously. In addition, the position of gene model would not influence the samples. To minimize the bias in selecting the polyA RNAs, we prepared the full transcript data and mature RNA data. In detail, mature RNA data exclude the sites on the intron region while the full transcript data covered either the exon or intron region.

![Alt text](https://github.com/yuxuanwu17/m6A_reader/blob/master/plot/m6a_criteria.png) 

**Figure 1** Criteria of determining the m6A reader binding sites.

### Deep learning model construction

Previously, we conducted a traditional machine learning about m6A reader by different encoding methods, however, the performance of one-hot method was not ideal, there was still a gap to improve. In addition, one-hot encoding method learned from the convolutional neural network (CNN) is suitable for learning potential motifs in the bioinformatics field, therefore, we opted for deep learning techniques in this research. Furthermore, recurrent neural network (RNN) was incorporated since it could capture the information in sequence, for instance, the potential relationship between each nucleotide. 

To build the deep learning model, we used Keras v2.3.0 and R v4.0.2 to conduct the learning part and process the raw data for prediction. For the data preparation part, we used R to extract n nucleotides (bp) of flanking sequences centered on the target adenosine, ranging from 251 to 2001bp to explore a suitable length. The processed sequence data were then inputted to Python3 for encoding, in this case, we chose One-hot encoding method for better model interpretability, for instance, A (1,0,0,0), C (0,1,0,0), G (0,0,1,0), T (0,0,0,1). The overall framework can be seen in Figure 2. Each sequence was then transformed to an n×4 matrix and fed into two combinations of 1D convolution (Conv1D) layer and max-pooling layer. For the first combination, we set 90 kernels with size equaled 5 and applied L2 regulation to prevent overfitting. The rectified linear unit (ReLU) was used as the activation function to provide our necessary non-linearity. The following max-pooling layer was set in size equaled 4 with strides 2 to reduce the dimension of output from the previous layer. The dropout rate was incorporated to 0.25 to further reduce the possibility of overfitting. A second 1D convolution (Conv1D) layer with 100 filters and size equaled 3 to extract the feature of the previous data. Similarly, the ReLU function and L2 regulation were applied. However, the max-pooling size was 10 with 1 stride, under which circumstance could the model achieve higher performance. 

The recurrent neural networks long short-term memory (LSTM) layer was used to aggregate the outputs of CNNs for predicting the RBP binding, in this case, the m6A readers’ substrates sites. LSTM processed sequentially of the sequence element, hoping to capture the inter-dependencies between motifs. Moreover, the fully connected layer with 1000 neurons would receive the output from the LSTM layer, and the non-linear activation function n, sigmoid, would calculate the prediction probability in each training class. The overall tuning process was used the loss function, binary cross-entropy to conduct the weight-tuning, optimizing the learning process, additionally, we found that Adam is the most suitable for this task. Finally, the output would be the probability of being m6A reader substrate sites. 

![Alt text](https://github.com/yuxuanwu17/m6A_reader/blob/master/plot/Architecture(3).png) 

**Figure 2** The sequence data are encoded by One-hot method and fed into the convolution layer and followed by the pooling layer twice to extract the significant features. The LSTM layer learns the long-term dependencies between sequence data generated by convolution layers. The flatten layer combines the previous kernels into a vector and inputs to the fully connected layer to calculate the probability of being m6A reader substrate site 

### Training strategy and performance evaluation

We separated each gene data set into three categories, training, testing, validation dataset, the ratio was 8:1:1 respectively. Moreover, to reduce the bias caused by imbalanced data samples, we ensured the same number of positive and negative samples in each category. The early stopping method was included to reduce the unnecessary computation during the learning process and the patience was designed as 10. The loss plot was drawn to document the training procedure and monitor the potential overfitting.  

To validate the model performance, four commonly used performance metrics, including area under the ROC curve (AUC), area under the Precision-Recall curve (PR-AUC), accuracy (ACC) and Mathew’s correlation coefficient (MCC). The formula of ACC and MCC are demonstrated as follows:

![Alt text](https://github.com/yuxuanwu17/m6A_reader/blob/master/plot/equation.png) 

where TP and TN are denoted as True Positive and True Negative, FN and FT are denoted as False Negative and False Positive. To sum up, the higher the performance metrics value, the more accurate the prediction. Additionally, we compared the performance with the previous research using machine learning method, the combination of CNN + RNN frameworks and the CNN framework only to determine the optimal choice.

We also exploited DeepExplain's epsilon-LRP method (gradient-based) to calculate the contribution in each feature input. With the assistance of this approach, we could rank the nucleotides’ significance in identifying the m6A readers’ substrates. Moreover, we extend the sequence upstream/downstream length from 50bp to 250bp, hoping to cover more information in determining each nucleotide contribution. 4 Results and discussion


## Results

### Performance comparison 

Classifiers might achieve varied performance on different datasets. To assess the fitness between the 6 reader binding site datasets and the two deep learning classifiers, models were built on full transcripts and their performance were analyzed. Similarly, different size of full transcripts was encoded with One-hot method. As shown in Figure 3(A), models using CNN classifier achieved theoretically good performance with overall AUROC larger than 0.8. It seems that the CNN classifier fit the YTHDF1 binding datasets better than other reader binding sites, with overall AUROC exceeding 0.9 and highest AUROC of 0.93. In addition, CNN model achieves good performance with YTHDF2 binding datasets as well, with highest AUROC of 0.929. It is noticeable that the performance of CNN models with EIF3a varied dramatically along with the size of transcript, from 0.96 to 0.81, which suggests that the performance of CNN classifier is depend on the size of transcripts. Similar trends can be seen in the YTHDC2 datasets, the trained model with different input size achieves different AUROC score, with optimal input transcript size of 251bp (AUROC = 0.89)

Regarding the fitness of CNN+RNN classifier with the six reader datasets, models shows similar performance for YTHDF3, YTHDF2, YTHDF1, YTHDC2 datasets (with AUROC around 0.9). Moreover, the model trained with these four datasets as well as the YTHDC1 datasets (with AUROC around 0.875) seems transcript-size independent since lines are relatively stable. Interestingly, the performance of models trained with EIF3a datasets varied greatly from length to length (AUROC varied from 0.88 to 0.94). The structure variation between YTH family protein and EIF3a might contribute to the difference on model performance.

To assess the feasibility of the two classifiers, namely CNN and RNN, performance of models was interpreted and compared. Figure 3(B) compares the performance of models with different size of EIF3a transcripts. As indicated, the combination of CNN and RNN classifier achieves overall better performance than the CNN classifier for both full transcript and mature transcript. Since the trend of line graph for CNN+RNN model is more stable than the line for CNN model, we can infer that the combination of CNN and RNN makes the model less dependent on the length of transcript used.

![Alt text](https://github.com/yuxuanwu17/m6A_reader/blob/master/plot/cmbd2.png)

**Figure 3** (A) Compared the performance of CNN model and CNN+RNN model in the prediction of six m6A reader substrates under different length in full transcripts. (B) Compared the AUROC value in either full transcript or mature transcript when predicting the EIF3A reader substrates. 

### ROC and PR-curve comparison between multiple sequence length

the Receiver Operating Characteristic (ROC) curve and Precision-Recall curve for EIF3A datasets under the combination of CNN and RNN were visualized in Figure 4. As can be seen from the figure that, although the performance was different under different sequence size, the overall trend was stable and devoid of fluctuation. In addition, the overall performance in full transcripts could outperform the mature transcripts, probably the reason that full transcript data could cover either the exon or intron region. 

Here, we mainly opted EIF3A dataset for easy demonstration in this paper. More details of the other five datasets could be achieved in the supplementary file.

![Alt text](https://github.com/yuxuanwu17/m6A_reader/blob/master/plot/rocs_update.png)

**Figure 4** (A) Compared the ROC curve and the regarding AUROC of mature and full transcript of EIF3A under CNN+RNN model in various sequence lengths. (B) Compared the PR curve and the regarding PRAUC of mature and full transcript of EIF3A under CNN+RNN model in various sequence lengths. 

### Quantify each input nucleotide contribution by the layer-wise relevance calculation

Each input feature was calculated to obtain its contribution to the results by DeepExplain’s epsilon-LRP method. The feature importance plots were based on the EIF3a binding site datasets (Figure 4). The higher score that the position gets, the larger probability that the center nucleotide is an EIF3a reader binding site if this nucleotide present at that position. As shown in the graph, positions located around the predicted m6A sites got significantly higher scores than other positions, which means those positions are more important in determining whether the center nucleotide is m6A reader substrate site or not. Additionally, the prediction of modification site would benefit from taking sequence more than 50bp upstream or downstream the predicted site since they include positions with high importance score

Specifically, a site would be less likely to be m6A modification site if the adenosine represents in 100bp downstream since the majority of position within this sequence got importance scores smaller than 0. In comparison, the presence of cytosine in 50 upstream/downstream the predicted site tends to boost the chance of the center nucleotide being modified. No specific patterns were found for guanine and thymine as the importance plot present a shape like the sine function. 

The results showed that if those positions 34bp, 59bp, 11bp, 58bp, 27bp, 49bp, 72bp upstream, 21bp, 27bp, 24bp, 25bp, 116bp downstream the modification site is cytosine, the site would more likely to be the EIF3a reader binding site. In addition, the probability of the modification site being EIF3a substrate site would decrease if guanosine was found on positions 21bp, 71bp, 33bp, 32bp, 31bp, 22bp upstream the center site or uridine was found on positions 54bp upstream or 53bp downstream the center site. The screened top 20 nucleotides that will decrease the change of the site being EIF3a modification site include: adenosines on positions 39bp, 27bp, 47bp, 61bp, 10bp, 12bp, 23bp, 170bp, 157bp, 51bp, 14bp, 226bp, 52bp upstream the center nucleotide, cytosine on positions 93bp upstream and 185bp downstream the center nucleotide, guanosine on positions 92, 97bp downstream the center site as well as uridines on positions 56bp, 63bp upstream the modification site.

![Alt text](https://github.com/yuxuanwu17/m6A_reader/blob/master/plot/contribution_plot.png)

**Figure 5** Feature importance scores in EIF3A full transcript prediction. We both extracted upstream/downstream 50bp and upstream/downstream 250 bp of the sequence to rank the contribution of each nucleotide in determining the binding site. In each position, the higher score it gains, the higher contribution towards the binding sites.

#How to use this file?

## Prerequisite packages
keras version 2.3.0, numpy, pandas, argparse

## Run 'python3 main.py -h' in command line for help

### Select the input genes
You could select 6 genes, including YTHDC1-2, YTHDF1-3, EIF3A

### Select the input length of the sequence
You could choose 251, 501, 1001, 2001bp input length to compare the prediction performance. 
Due to the size limitation, i only upload some some small sized sequence as examples.
Email me, if you want to apply all the data

### Select the condition of the sequence, either full transcript (full) or mature RNA (exon)
To minimize the bias in selecting the polyA RNAs, we prepared the full transcript data and mature RNA data. In detail, mature RNA data exclude the sites on the intron region while the full transcript data covered either the exon or intron region.

### Choose the CNN or CNN+RNN model
To compare the performance under different framework
