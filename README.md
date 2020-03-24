# Find Significant Genes for Tumor Classification from a Convolutional Neural Network (CNN)

The interpretability of a model is usually more important than prediction accuracy in analyzing biological data. Deep CNN model was deemed as a powerful tool for image classification which at the same time lacks interpretability. However there are more and more tools have being developed to fill the gap.

By using guided Grad-CAM, we can identify pixels that are important for image classification, which have enabled us to loook into the 'black box' of deep neural networks and understand how it works. 

Here I downloaded a large gene expression dataset from TCGA (https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga), and embeded the data into 2D images, then trained a CNN model on it. The model accurately classified the 22 tumor types, and Grand-CAM also worked perfectly for tumor-specific gene identification as you can see in 'granCAM.png'.

The next step would be validating the selected genes in an independent dataset... 

