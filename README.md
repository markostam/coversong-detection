# coversong-detection
A cover song, cover version, or simply cover, by definition, is a new performance or recording of a previously recorded, commercially released song. It may be by the original artist themselves or a different artist altogether. Automatic cover song detection has been an active research area in the field of Computer Audition for the past decade. In this paper, we propose a novel method for cover song detection using automatic extraction of audio features with a stacked auto-encoder (SAE) combined with beat tracking in order to maintain temporal synchronicity.

[Paper describing method](http://www.ece.rochester.edu/~zduan/teaching/ece477/projects/2015/Marko_Stamenovic_ReportFinal.pdf)

[Poster describing method](http://www.ece.rochester.edu/~zduan/teaching/ece477/projects/2015/Marko_Stamenovic_Poster.pdf)

#Main Functions

##trainSAE_allsongs
Main system function called to train SAE on all original songs. Extracts features, calls secondary functions for SAE training.
In this case we have use two different functions for CQT and Chroma input sized patches
(trainSAE_allsongs_cqt = CQT features, trainSAE_allsongs_chroma = chroma features)

##test_final_allsongs
Main system function to classify cover songs as covers or not. Extracts features, calls secondary functions for SAE testing.
In this case we have use two different functions for CQT and Chroma input sized patches
(test_final_allsongs_cqt = CQT features, test_final_allsongs_chroma = chroma features)

#Secondary Functions

##autoencoderCost.m
This function calcualtes the overall cost of an auto-encoder on all input
data, and the partial derivatives of the cost w.r.t all weights.

##forwardActivation.m
Forward calculation of auto-encoder on a set of input examples.
The input and output layer each have n nodes and the hidden layer
has m nodes.

##lbfgsFunc.m
Calls the well-known L-BFGS function for autoencoder optimization

##sampleIMAGES_AftRemv
Reorders data into correct patch sizes and normalizes for input into autoencoder
In this case we have use two different functions for CQT and Chroma input sized patches
(sampleIMAGES_AftRemv2 = CQT, sampleIMAGES_AftRemv_chroma = chroma)

##tool
Folder containing various open source functions used in feature extraction and feature distance calculation including DTW, CQT, beat tracking, etc. Many come from [LabROSA](http://labrosa.ee.columbia.edu).

##minFunc
Folder containing various open source algorithms for neural net training. In this case we use L-BFGS.

##chroma-ansyn
Folder containing open source algorithm for chroma feature extraction from [LabROSA](http://labrosa.ee.columbia.edu).

#Dataset
##cover_seeds
Original songs used to train autoencoder

##cover_pairs
All original songs + their cover pairs.
