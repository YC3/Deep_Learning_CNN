# Find Significant Genes for Tumor Classification from a Convolutional Neural Network (CNN)

A large gene expression dataset was downloaded from TCGA (https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga), then the expression profile of single samples was embedded into 2D images, next a CNN model was trained on the training set and validated in the blind test set. The model accurately classified the 22 tumor types, and Grad-CAM also worked perfectly for tumor-specific gene identification as you can see in 'granCAM.png'.

The next step would be validating the selected genes in an independent dataset... 
