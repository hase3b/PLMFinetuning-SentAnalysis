# PLMFinetuning-SentAnalysis
This repository contains all the code, results, and documentation for our CSE674: Text Analytics term project. The objective was to explore and compare classical machine learning methods with fine-tuned pre-trained language models (PLMs) using Low-Rank Adaptation (LoRA) for sentiment analysis on the IMDb dataset. This project is a collaborative effort of Abdul Haseeb, Annayah Usman, and Sawera Hanif.

## **Overview**
This project investigates the effectiveness of parameter-efficient fine-tuning techniques using LoRA compared to classical machine learning approaches for sentiment classification as well as a baseline fine-tuned PLM. The analysis involves:

1. Classical ML Models: Models like Logistic Regression, Naive Bayes, and k-NN trained on TF-IDF and Word2Vec embeddings.
2. Fine-Tuned PLMs: PLMs such as DistilBERT, RoBERTa, ALBERT, and GPT2 fine-tuned with LoRA to optimize resource usage while maintaining high performance.

## **Results Summary**
Baseline Accuracy: 0.89 (Distilbert-base-uncased-finetuned-sst-2-English).
Best Classical ML Model: Logistic Regression with 0.89 accuracy.
Best Fine-Tuned PLM: RoBERTa surpassing 0.91 accuracy and F1-score, outperforming all other models.
LoRA fine-tuning allowed training <1% of PLM weights while achieving exceptional performance.
For more details, refer to the main report titled "Text_A3_Report_Annayah_AbdulHaseeb_Sawera.pdf".

## **Repo Structure**
├── EDA  
│   └── Contains exploratory data analysis (EDA) plots and notebook.  
├── Misc Scripts  
│   └── Notebooks that were not part of the main study but provide additional insights.  
├── Results  
│   └── CSV files with detailed experimentation results and final fine-tuning outcomes.  
├── Scripts - Classical ML Models + Baseline PLM  
│   ├── ClassicalMLModels_TFIDF_Embeddings.ipynb  
│   ├── ClassicalMLModels_TFIDF_Hyperparams.ipynb  
│   └── FT_Baseline.ipynb  
├── Scripts - Experimentation PLMs  
│   ├── Finetuning Hyperparameter Experimentation notebooks for ALBERT, DistilBERT, GPT2, and RoBERTa.  
├── Scripts - Final Finetuned PLMs  
│   └── Final fine-tuning notebooks for the optimal configurations for each model.  
├── Assignment3.pdf  
│   └── Contains the project task details.  
└── Text_A3_Report_Annayah_AbdulHaseeb_Sawera.pdf  
    └── The comprehensive project report.  

## **Key Insights**
* Classical ML Models:
 * Logistic Regression performed on par with the baseline PLM at 0.89 accuracy.
 * k-NN showed significant improvement with Word2Vec embeddings, reaching 0.78 accuracy.

* Fine-Tuning Experiments:
 * Phase 1: Training hyperparameters (e.g., batch size, learning rate, epochs) significantly influenced performance. Higher learning rates (e.g., 0.0001) and batch sizes (8, 16) yielded optimal results.
 * Phase 2: Targeting fewer matrices (e.g., Query matrix in attention layers) often achieved comparable accuracy to targeting all matrices, demonstrating LoRA's efficiency.

* Final Results:
 * All fine-tuned PLMs surpassed the baseline and classical ML models.
 * RoBERTa consistently achieved the highest performance across metrics.
 * Fine-tuning GPT2 on just 5% of the dataset yielded surprising accuracy (0.91), indicating potential with larger datasets.

## **Future Work**
* Ensemble Models: Combine the fine-tuned models pushed to HuggingFace into an ensemble for improved classification.
* Hyperparameter Tuning: Develop a systematic methodology for better guidance on optimal configurations.
* Adaptive Rank Selection: Optimize rank tradeoff for performance and computational efficiency in LoRA.
* Extended Training: Train models for more epochs to observe potential improvements.

## **Acknowledgments**
We would like to acknowledge the following tools, models, and resources that contributed to this project:
* Tools: PyTorch, HuggingFace Transformers, Scikit-learn, Matplotlib, Seaborn, Pandas, Numpy.
* Models: DistilBERT, RoBERTa, ALBERT, GPT2 (pre-trained models fine-tuned with LoRA).
* Data Source: IMDb Dataset for Sentiment Analysis (sourced via HuggingFace Datasets).
* Instructor: Dr Sajjad Haider (Professor IBA Karachi)
