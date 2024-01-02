# DSS_Master_Thesis

This repository hosts the code used for my Master Thesis.
The main goal of the thesis was to predict commercial flight delays in the U.S. using machine and deep learning models.
A overview image can be found below that displays the diferent fases, methods and steps used in this project.
The files are organized based on the colours of each fase.

![alt text](https://github.com/rvloenhout/DSS_Master_Thesis/blob/main/Methods_overview.png?raw=true)

Below you will also find the abstact of the thesis:

Air travel stands out as one of the most widely embraced means of transportation between continents and major cities. 
Nevertheless, delays significantly impact travelers’ experiences, incur costs for airlines, and contribute to environmental emissions. 
This thesis endeavours to mitigate these delays through predictive modelling, utilizing a combination of U.S. flight, aircraft (BTS), weather (NOAA), and demographic data (USCB). 
The central research question guiding this study is: *To what extent can U.S. flight delays be predicted on imbalanced data using Machine/Deep learning models, and how do airlines differ in terms of the predictability of delays?*
The problem is framed as a binary classification task, focusing on predicting whether a delay has occurred.

In contrast to existing literature, a noteworthy aspect of this thesis is the application of TabNet to the problem. 
The objective is to assess whether a more complex model can outperform established models. 
Additionally, recognizing the imbalanced nature of the data, three resampling techniques (over-, under-, and hybrid resampling) are explored to evaluate their impact on performance. The study identifies XGBoost, employing a hybrid resampling technique, as the most accurate solution for the problem, achieving an AUROC score of 0.819. 
Although TabNet’s performance fell below expectations, an engineered feature examining whether a previous flight was delayed emerged as a significant predictor of delays. 
Lastly, a distinct variation in prediction errors among airlines was observed, with Hawaiian Airlines proving the least predictable and Frontier Airlines the most predictable.