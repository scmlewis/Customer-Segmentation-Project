# Customer-Segmentation-Project

# Customer Segmentation for Personalized Financial Product Offerings
MSc Business Analytics project at HKUST Business School, using Kaggle’s Bank Marketing dataset (https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset).

## Data
- Source: Kaggle Bank Marketing dataset (11,162 records, 17 features).
- Features: age, job, balance, deposit, pdays, etc.

## Notebooks
- `data_exploration.ipynb`: Initial exploration.

## Visualizations
- `age_distribution.png`, `balance_distribution.png`, `deposit_counts.png`: Exploration plots.
![deposit_counts](https://github.com/user-attachments/assets/86447702-dd9b-49a6-b02d-1f53e4960b95)
![balance_distribution](https://github.com/user-attachments/assets/ea509297-2a1f-485b-b833-721ad275ac68)
![deposit_counts](https://github.com/user-attachments/assets/cb772f8b-1c45-41df-96da-eb92d3eea54e)


- `pca_4_clusters.png`: PCA scatter plot with 4 clusters.
![pca_4_clusters](https://github.com/user-attachments/assets/4bdc4a4c-6cb0-4d68-90b3-580c6e014a7b)


## 1. Introduction
This report presents the findings and recommendations from a customer segmentation analysis conducted as part of the MSc in Business Analytics at HKUST Business School. Utilizing Kaggle’s Bank Marketing dataset (11,162 records, 17 features), the project aimed to segment customers into 4–6 clusters, characterize these segments, and propose tailored marketing strategies to achieve a projected 12–18% conversion increase. The analysis leverages clustering techniques to enhance personalized financial product offerings.

## 2. Methodology
### 2.1 Data Acquisition
The dataset was sourced from Kaggle (https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset), containing demographic (age, job), financial (balance), and behavioral (pdays, deposit) features. It was saved to `\data\bank.csv` and explored using Python (pandas, seaborn) to confirm 11,162 rows, no missing values, and balanced deposit distribution.

### 2.2 Data Preprocessing
- **Outlier Handling**: `balance` was capped at the 99th percentile (~13226.98) to address extreme values, and `pdays` (-1 indicating no prior contact) was replaced with 0.
- **Feature Engineering**: Numeric features (`age`, `balance`, `pdays`) were standardized using `StandardScaler`, and categorical features (`job`, `marital`, `education`, `housing`, `loan`, `deposit`) were one-hot encoded with `drop='first'` using `OneHotEncoder`, resulting in a preprocessed shape of (11162, 22).
- **Tools**: Python with scikit-learn.

### 2.3 Clustering Analysis
- **Algorithm**: K-means clustering was applied with a random state of 42 for reproducibility.
- **Cluster Selection**: The Elbow Method indicated a bend at 5 clusters, but a Silhouette Score of 0.1894 for 4 clusters outperformed 0.1537 for 5 clusters, favoring 4 clusters for better cohesion.
- **Output**: Clusters were profiled using `age`, `balance`, `pdays`, and `deposit_numeric` (converted from 'yes'/'no' to 0/1).

### 2.4 Visualization
- **Tool**: Matplotlib was used to generate a PCA scatter plot, saved as `pca_4_clusters.png`, visualizing the 4-cluster solution.
- **Data Export**: Clustered data was saved as `clustered_data.csv` for potential future analysis.

## 3. Results
### 3.1 Cluster Profiles
- **Cluster 0**: 3277 customers, age 54.79, balance 1120.16, pdays 23.56, deposit_numeric 0.474 (older, moderate balance, balanced interest).
- **Cluster 1**: 753 customers, age 43.63, balance 8417.34, pdays 44.02, deposit_numeric 0.566 (middle-aged, high balance, moderate interest).
- **Cluster 2**: 1460 customers, age 38.20, balance 1021.00, pdays 286.33, deposit_numeric 0.581 (younger, moderate balance, high pdays, higher interest).
- **Cluster 3**: 5672 customers, age 33.86, balance 780.03, pdays 9.32, deposit_numeric 0.434 (youngest, lowest balance, low interest).
- **Visualization**: The PCA plot confirms four distinct segments with reasonable separation.

## 4. Recommendations
### 4.1 Marketing Strategies
- **Cluster 0**: Promote retirement savings plans (e.g., annuities) and low-risk investments via direct mail and phone calls. Launch a quarterly campaign from July 1, 2025, with 30-day follow-ups. Budget: 25%, Metrics: 15% response, 5% conversion in 90 days.
- **Cluster 1**: Offer premium investments (e.g., stock portfolios) and insurance via email with tools and phone consultations. Start a bi-monthly campaign from August 1, 2025, with 15-day slots. Budget: 30%, Metrics: 20% response, 8% conversion in 60 days.
- **Cluster 2**: Provide re-engagement incentives (e.g., 2% bonus interest) via email and SMS with an online portal. Begin a monthly campaign from June 15, 2025, with weekly SMS for 30 days. Budget: 25%, Metrics: 18% response, 6% conversion in 45 days.
- **Cluster 3**: Offer introductory products (e.g., no-fee savings) via social media ads and a mobile app. Initiate a continuous campaign from July 1, 2025, with bi-weekly refreshes. Budget: 20%, Metrics: 15% response, 4% conversion in 90 days.

### 4.2 Projected Impact
The targeted strategies are projected to increase conversion rates by 12–18%, with segment-specific metrics contributing to a cumulative uplift based on response and conversion rates.

## 5. Conclusion
This analysis successfully segmented 11,162 customers into four clusters, providing actionable insights for personalized marketing. The recommendations, supported by a Silhouette Score of 0.1894, aim to enhance customer engagement and conversion. Next steps include finalizing the report and hosting deliverables.
