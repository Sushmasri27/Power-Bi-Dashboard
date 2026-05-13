#RetailPulse -- Project Report
What is RetailPulse?
RetailPulse is an AI-powered analytics platform that helps retail businesses make smarter decisions. It takes raw sales transaction data and turns it into actionable insights:

Who are your best customers? (Customer Segmentation)
How much will you sell next month? (Demand Forecasting)
Which customers might leave? (Churn Prediction -- upcoming)
How much stock should you keep? (Inventory Optimization -- upcoming)
The Dataset
We used the Online Retail II dataset from the UCI Machine Learning Repository -- a real-world dataset from a UK-based online gift retailer.

Property	Value
Source	UCI ML Repository
Total rows	525,461 transactions
Time period	December 2009 -- December 2010
Customers	4,383 unique
Products	4,632 unique
Countries	40
Raw columns (what we started with):
Column	Type	What it means	Example
Invoice	text	Unique bill number. If it starts with "C", it's a cancelled order	489434, C489449
StockCode	text	Product code	85048, 22423
Description	text	Product name	"15CM CHRISTMAS GLASS BALL 20 LIGHTS"
Quantity	number	How many units were bought in this line item	12, 6, 48
InvoiceDate	date/time	When the purchase happened	2009-12-01 07:45:00
Price	number	Price per unit in British Pounds	6.95, 1.25
Customer ID	number	Unique ID for each customer (sometimes missing)	13085, NaN
Country	text	Where the customer is from	United Kingdom, France
Step-by-Step Process
Day 1: Exploratory Data Analysis (EDA)
Goal: Understand the raw data before doing anything with it.

What is EDA? It's like opening a new puzzle box and looking at all the pieces before you start assembling. You want to know: How many pieces are there? Are any broken? What patterns do you see?

What we did:
Step	What we checked	What we found
Shape check	How big is the data?	525,461 rows, 8 columns
Missing values	Any empty cells?	20.5% of Customer IDs are missing (~107K rows)
Data types	Are numbers stored as numbers?	Yes, but Customer ID is float (should be int)
Distribution analysis	What do typical values look like?	Most orders are small (1-12 units), most prices are under 5 pounds
Outlier detection	Any extreme values?	Some quantities in the thousands, some negative (returns)
Time patterns	When do people buy?	Tuesdays and Thursdays are busiest. November peak (holiday shopping)
Top products	What sells most?	Gift wrap, lunch bags, and decorative items dominate
Country breakdown	Where are customers?	90%+ from UK
Figures generated:
Missing values heatmap
Distribution plots (quantity, price, revenue)
Box plots for outlier detection
Monthly and daily sales trends
Day-of-week and hourly patterns
Correlation heatmap
Top products/countries bar charts
Day 2: Data Cleaning & Feature Engineering
Goal: Remove garbage data and create new useful columns.

Part A: Cleaning
Think of this like filtering dirty water. We remove things that would pollute our analysis:

What we removed	Why	Rows affected
Cancelled invoices (starting with "C")	These are returns, not sales	~9,000
Missing Customer IDs	Can't segment a customer we don't know	~107,000
Price <= 0	Free items or errors	~few hundred
Quantity <= 0	Returns already caught above, or errors	~few hundred
Exact duplicate rows	Data entry errors	~18,000
Result: 525,461 rows → ~390,000 rows (74% retention)

New column added:

Revenue = Quantity x Price (how much money each line item generated)
Part B: RFM Feature Engineering
What is RFM? It's a marketing framework that scores every customer on three dimensions:

Metric	What it measures	How we calculated it
Recency	How recently did they buy?	Days between their last purchase and our reference date
Frequency	How often do they buy?	Count of unique invoices per customer
Monetary	How much do they spend?	Total revenue per customer
Scoring (1-5): We used pd.qcut (quantile-based binning) to split each metric into 5 equal-sized groups:

Score 5 = best (most recent, most frequent, highest spending)
Score 1 = worst
Example:

Customer	Recency (days)	R Score	Frequency	F Score	Monetary	M Score	Total RFM
Alice	3 days ago	5	15 orders	5	5,000 pounds	5	15
Bob	200 days ago	1	1 order	1	50 pounds	1	3
Segment labels were assigned based on score patterns:

Segment	Rule	What it means
Champions	R>=4, F>=4, M>=4	Best customers -- buy often, spend a lot, bought recently
Loyal Customers	R>=3, F>=3, M>=3	Consistent and valuable
New Customers	R>=4, F<=2	Just started buying -- nurture them
At Risk	R<=2, F>=3, M>=3	Were good customers but haven't bought recently
Can't Lose Them	R<=2, F<=2, M>=3	High spenders who went silent -- urgent attention needed
Lost	R<=2, F<=2, M<=2	Haven't bought in a long time, low value
Part C: Daily Sales Features
We also aggregated transactions into one row per day with rolling statistics:

Feature	What it is	Why it matters
total_revenue	Sum of all revenue that day	The main target for forecasting
total_quantity	Total items sold	Volume indicator
transaction_count	Number of orders	Activity indicator
unique_customers	How many different people bought	Customer base health
revenue_ma_7d	Average revenue over last 7 days	Smooths out daily noise
revenue_ma_30d	Average revenue over last 30 days	Shows the longer trend
revenue_std_7d	How much revenue varies over 7 days	Measures volatility
revenue_lag_1d	Yesterday's revenue	For the model to learn "what happened yesterday"
revenue_lag_7d	Revenue from 7 days ago	For weekly patterns
day_of_week	0=Monday, 6=Sunday	Captures weekly buying habits
is_weekend	1 if Sat/Sun, 0 otherwise	Weekend vs weekday effect
Day 3: Customer Segmentation with Machine Learning
Goal: Group customers into meaningful segments using clustering algorithms (not manual rules).

Algorithm 1: K-Means
How K-Means works (simplified):

Pick K random points as "cluster centers"
Assign every customer to their nearest center
Move each center to the average position of its assigned customers
Repeat steps 2-3 until centers stop moving
Before clustering, we preprocessed the data:

Log-transform: Revenue is heavily skewed (few big spenders, many small). Log squashes the long tail.
StandardScaler: Makes all features have mean=0 and std=1, so no feature dominates by scale.
Finding the right K:

Method	What it does	Result
Elbow method	Plot inertia vs K, find the "bend"	Shows diminishing returns
Silhouette Score	Measures how well-separated clusters are (-1 to 1)	Higher = better
Calinski-Harabasz	Ratio of between/within cluster variance	Higher = better
We tested K=2 through K=10 and picked the K with the highest silhouette score.

Algorithm 2: DBSCAN
How DBSCAN works (simplified):

Instead of pre-defining K, it finds "dense regions" of points
Points in sparse areas are labeled as "noise" (outliers)
Two parameters: eps (neighborhood radius) and min_samples (minimum neighbors to form a cluster)
Key difference from K-Means: DBSCAN can find oddly-shaped clusters and automatically detects outliers.

Tuning: We used a k-distance graph to find a good eps, then tested multiple combinations via grid search.

Day 4: Time-Series Preparation
Goal: Get the daily revenue data ready for forecasting models.

Stationarity Testing
What is stationarity? A time series is "stationary" if its average and spread don't change over time. Most forecasting models need stationary data.

Tests we ran:

Test	Null Hypothesis	How to read it
ADF (Augmented Dickey-Fuller)	"The series is non-stationary"	p < 0.05 → reject → it IS stationary
KPSS	"The series IS stationary"	p < 0.05 → reject → it is NOT stationary
We ran both tests because using just one can be misleading.

Seasonal Decomposition
We broke the revenue series into three components:

Component	What it captures
Trend	Long-term direction (is revenue going up or down over months?)
Seasonal	Repeating patterns (e.g., every Tuesday is high, every Sunday is low)
Residual	Random noise left after removing trend and seasonality
We did this at two periods:

Weekly (period=7): Captures day-of-week patterns
Monthly (period=30): Captures month-level cycles
Differencing
If the series wasn't stationary, we applied differencing:

First-order (d=1): Subtract yesterday's value → removes trend
Seasonal (d=7): Subtract value from 7 days ago → removes weekly pattern
ACF/PACF Analysis
These plots show how today's value correlates with past values:

ACF (Autocorrelation): Correlation with all past lags
PACF (Partial Autocorrelation): Correlation with a specific lag, removing the effect of intermediate lags
These help identify the right parameters for forecasting models.

Day 5: Prophet Forecasting
Goal: Build a demand forecasting model using Facebook Prophet.

What is Prophet? A time-series forecasting model built by Facebook/Meta. It automatically handles trends, seasonality, and holidays.

Train/Test Split
Set	Size	Period	Purpose
Train	344 days	Dec 2009 -- Nov 2010	Model learns from this
Test	30 days	Nov -- Dec 2010	We check accuracy on this
Baseline Model
Default Prophet parameters
Weekly seasonality only
Computed MAPE (Mean Absolute Percentage Error) as the baseline
What is MAPE? It tells you "on average, how far off is your prediction as a percentage of the actual value." A MAPE of 10% means your predictions are typically within 10% of reality.

Tuned Model
We tested 24 parameter combinations:

Parameter	What it controls	Values tested
changepoint_prior_scale	How flexible the trend line is	0.01, 0.05, 0.1, 0.5
seasonality_prior_scale	How strong the seasonal patterns are	0.1, 1.0, 10.0
seasonality_mode	How seasonality combines with trend	additive, multiplicative
We also added monthly seasonality and UK public holidays.

30-Day Future Forecast
After finding the best parameters, we retrained on ALL data and predicted 30 days into the future.

Day 6: LSTM Forecasting with PyTorch
Goal: Build a neural network model for forecasting using LSTM.

What is LSTM? Long Short-Term Memory -- a type of neural network designed to learn from sequences. It "remembers" patterns from the past to predict the future.

Architecture
Input (30 days x 7 features)
    ↓
LSTM Layer 1 (64 hidden units)
    ↓
LSTM Layer 2 (64 hidden units)
    ↓
Dropout (20% -- randomly turns off neurons to prevent overfitting)
    ↓
Fully Connected Layer → 1 output (next day's revenue)
Setting	Value	Why
Lookback window	30 days	Give the model a month of history to learn from
Features	7	Revenue, quantity, transactions, customers, day_of_week, is_weekend, month
Hidden units	64	Balance between capacity and overfitting
Layers	2	Deeper than 1 for better pattern learning
Optimizer	Adam	Adaptive learning rate, industry standard
Early stopping	Patience=15	Stop if loss doesn't improve for 15 epochs
Gradient clipping	max_norm=1.0	Prevents exploding gradients
Training Process
Normalize all features to [0, 1] range using MinMaxScaler
Create sliding windows: days 1-30 → predict day 31, days 2-31 → predict day 32, etc.
Train in batches of 32, shuffled
Save the best model weights when loss improves
Stop early when the model stops improving
Day 7: MLflow Experiment Tracking
Goal: Log all model experiments into MLflow for reproducibility.

What is MLflow? A tool that tracks your machine learning experiments -- it records what parameters you used, what metrics you got, and saves the model artifacts so you can reproduce results later.

What we logged:
Run	Parameters logged	Metrics logged
K-Means Segmentation	algorithm, n_clusters, features, scaling	segment counts, total customers
Prophet Forecasting	model, seasonality settings, holidays, test_days	MAPE, MAE, RMSE
LSTM Forecasting	model, layers, hidden_dim, dropout, lookback, LR	MAPE, MAE, RMSE
Also generated a 6-panel summary dashboard figure covering all Week 1 work.

Day 8: Hybrid Ensemble (Prophet + LSTM)
Goal: Combine Prophet and LSTM into a single, better model.

Why ensemble? Different models have different strengths. Prophet is good at capturing seasonality and holidays. LSTM is good at learning complex non-linear patterns. Blending them often beats either one alone.

4 Ensemble Methods Tested:
Method	How it works
Simple Average	prediction = (Prophet + LSTM) / 2
Weighted Average	Weight each model inversely proportional to its error. Better model gets more weight.
Optimal Blend	Grid search 21 weight combinations (0%, 5%, 10%...100% for Prophet) and pick the one with lowest MAPE
Linear Stacking	Train a linear regression on top of both models' predictions to find optimal combination
How we compared:
All 6 models (2 individual + 4 ensemble) were evaluated on the same 30-day test set using:

MAPE -- percentage error (lower is better)
MAE -- average absolute error in pounds (lower is better)
RMSE -- penalizes large errors more (lower is better)
The best-performing model was selected automatically.

Data Flow Summary
Raw Excel File (525K rows, 8 columns)
    │
    ├── [Day 1] EDA → understand patterns, distributions, missing values
    │
    ├── [Day 2] Cleaning → remove cancellations, missing IDs, bad prices
    │                       (525K → 390K rows)
    │
    ├── [Day 2] Feature Engineering
    │       │
    │       ├── Customer-level: RFM scores → customer_rfm.csv (4,312 × 10)
    │       │
    │       └── Daily-level: rolling stats, lags → daily_sales_features.csv (374 × 24)
    │
    ├── [Day 3] Clustering
    │       │
    │       └── K-Means + DBSCAN on RFM → customer_segments.csv (4,312 × 11)
    │
    ├── [Day 4] Time-Series Prep
    │       │
    │       ├── Stationarity tests (ADF, KPSS)
    │       ├── Seasonal decomposition
    │       ├── prophet_ready.csv (374 × 2)
    │       └── lstm_ready.csv (374 × 8)
    │
    ├── [Day 5] Prophet → prophet_forecast_30d.csv
    │
    ├── [Day 6] LSTM → lstm_predictions.csv + lstm_best.pt
    │
    ├── [Day 7] MLflow logging of all experiments
    │
    └── [Day 8] Ensemble → ensemble_predictions.csv + model_comparison.csv
Files Generated
Processed Data (data/processed/)
File	Rows	Columns	Description
customer_rfm.csv	4,312	10	RFM scores and segment labels per customer
customer_segments.csv	4,312	11	RFM + K-Means and DBSCAN cluster assignments
daily_sales_features.csv	374	24	Daily aggregates with rolling stats and lags
prophet_ready.csv	374	2	Date + revenue for Prophet
lstm_ready.csv	374	8	Date + 7 features for LSTM
prophet_forecast_30d.csv	30	4	30-day future forecast
prophet_forecast_full.csv	404	7	Full historical + future forecast
lstm_predictions.csv	30	4	LSTM test set predictions
ensemble_predictions.csv	30	8	All model predictions on test set
model_comparison.csv	6	4	Final model ranking
Figures (reports/figures/)
32 visualization files covering EDA distributions, time-series decomposition, cluster analysis, forecast comparisons, and summary dashboards.

Models (models/)
File	Description
lstm_best.pt	Best LSTM model weights (PyTorch)
