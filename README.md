## ML Projects

#### Technical Skills: Python, SQL, MATLAB

### Image Translation (SAR to EO) 
[Report](image_translation/Image_Translation_Report.pdf) | [Code](image_translation/Image_Translation_Final.ipynb) 

- Used Pix2Pix GAN framework to translate images from SAR (Synthetic Aperture Radar) to EO (Electro-Optical)
- Utilized the Spacenet 6 dataset (publicly available) containing satellite imagery (pair of 10000 SAR and Optical)




### Information Retrieval 
[Report](information_retrieval/NLP_TermProject_Final.pdf) | [Code](information_retrieval/SUBMISSION_NLP_IR_Final_NB.ipynb) 

- Implemented NLP techniques on Cranfield dataset to retrieve the top 10 relevant documents given a query
- Used tfidf vectorization based on Vector Space Model & 2 variants of Latent Semantic Indexing to compute similarity between a query & document
- Improved mAP & nDCG score in the retrieval process by 30% compared to the baseline Vector Space Model 

### Song Recommender System
[Report](recommender_system/RecommenderSystem_WriteUp.pdf) | [Code](recommender_system/RecommenderSystem_LIghtGBM.ipynb) 
- Built ML model to predict the user rating for a song given song ID, user ID, genre, etc. with 1.3M data points
- Performed data pre-processing; implemented an ensemble of collaborative filtering and LightGBM regressor 
- Trained the model on 700k data points and obtained rmse score of 0.738 on test data (5th among 45 teams)


### Propensity Prediction 
[Report](propensity_prediction/PropensityModel_FinalProjectReport.pdf) | [Code](propensity_prediction/PropensityPrediction_EDA.ipynb) 
- Developed a supervised ML model to predict the customer propensity to apply for the new line of credit card
- Achieved an AUC score of 0.752 (3rd out of 15 teams) using XGBoost; benchmarked it against Random Forest 


### Anamoly Detection
[Report](anomaly_detection/CriticalReview-BayesianOnlineChangepointDetection.pdf) | [Code](anomaly_detection/BayesianChangePoint.m) 
- Implemented Bayesian Change-Point Detection in MATLAB to detect abrupt changes in the time-series data
- Extended for the case of detecting a change in an AR (Auto Regressive) process by employing an RLS (Recursive Least Squares) estimator algorithm

