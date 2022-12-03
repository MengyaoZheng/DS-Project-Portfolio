# DS Project Portfolio

## Mengyao Zheng [here](https://www.linkedin.com/in/mengyao-zheng/).



### 1. Causal Inference on the average treatment effect (ATE) of quitting smoking ($T$) on weight gain ($Y$)
I have performed a causal analysis on a real-world healthcare dataset, known as the *NHANES I Epidemiologic Follow-up Study (NHEFS)* dataset. It is a government-initiated longitudinal study designed to investigate the relationships between clinical, nutritional, and behavioral factors. For more detail, please see [here](https://wwwn.cdc.gov/nchs/nhanes/nhefs/default.aspx/).

The task is to estimate the average treatment effect (ATE) of quitting smoking ($T$) on weight gain ($Y$). The NHEFS cohort includes 1,566 cigarette smokers between 25 - 74 years of age who completed two medical examinations at separate time points: a baseline visit and a follow-up visit approximately 10 years later. Individuals were identified as the treatment group if they reported smoking cessation before the follow-up visit. Otherwise, they were assigned to the control group. Finally, each individual’s weight gain, $Y$, is the difference in *kg* between their body weight at the follow-up visit and their body weight at the baseline visit. 

The notebook includes the following parts:
1. how a mechanism of how confounders, when unadjusted, can introduce bias into the ATE estimate. 
2. Implement propensity score re-weighting to estimate the ATE in Python.
3. Implement covariate adjustment strategies to estimate the conditional average treatment effect (CATE) as well as ATE in Python.
4. Assess how robust the ATE estimate is against potential unobserved confounders via sensitivity analysis. 

## NLP: Context-Aware Legal Case Citation Prediction Using Deep Learning [Presentation](https://www.youtube.com/watch?v=QfXUCw_XsT4)
We have craped 100K+ legal texts from Harvard Law School’s database using Python Scrapy, cleaned and tokenized. And we predicted in-text citation by both supervised and unsupervised learning methods: 1) building LSTM and CNN models with embedding layers using TensorFlow, 2) leveraging Legal-BERT model to obtain the embeddings and then used FAISS index to do similarity-based modelling. The Top model (LSTM) achieved a 200x accuracy boost as compared to the baseline random model.


