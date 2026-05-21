# Market-Basket-Analysis

# 🛒 Market Basket Analysis & Sales Optimization

## 📌 Business Problem
**Kitabi Duniya**, a heritage bookstore, experienced a collapse in annual growth due to the rise of online selling. The objective of this project is to utilize Machine Learning to discover actionable buying patterns, with the goal of increasing customer footfall and maximizing sales revenue by an expected 25%.

By applying **Market Basket Analysis (Apriori Algorithm)**, this project identifies highly correlated book categories to drive physical store layout changes, targeted bundle discounts, and online cross-selling strategies.

## 🛠️ Technical Stack
* **Language:** Python
* **Machine Learning:** `mlxtend` (Apriori, Association Rules)
* **Data Manipulation:** `pandas`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Database Pipeline:** MySQL, `sqlalchemy`, `pymysql`

---

## 🚀 Step-by-Step Project Workflow

### Step 1: Database Integration & Data Ingestion
To simulate a production environment, the raw data is passed through a relational database rather than just reading a static CSV:
* Read the 2,000 retail transactions (covering 11 different book categories) from a CSV file.
* Used `SQLAlchemy` to establish a connection to a local MySQL database.
* Pushed the raw transactional dataset into a SQL table named `book_store`.
* Queried the database to pull the data back into a Pandas DataFrame for modeling.

### Step 2: Data Pre-Processing
* The dataset consists of a binary transactional matrix (0 = not purchased, 1 = purchased).
* Converted the binary integer formats into explicit `boolean` types to optimize the Apriori algorithm's memory usage and processing speed.

### Step 3: Mining Frequent Itemsets (Apriori)
* Implemented the **Apriori Algorithm** to find frequent combinations of books bought together.
* **Support Threshold:** Set `min_support = 0.1` (10%) to ensure only statistically significant, high-traffic book combinations were analyzed, eliminating random noise.

### Step 4: Generating Association Rules & Data Cleaning
* Extracted actionable association rules using **Lift** as the primary metric (threshold > 1.2), ensuring the relationships are genuinely dependent, not just coincidental.
* **Feature Engineering for Clean Data:** Created a custom lambda function using `frozenset` to combine 'Antecedents' and 'Consequents'. This successfully identified and dropped redundant/reversed duplicate rules (e.g., {CookBks -> ChildBks} and {ChildBks -> CookBks}), keeping only the strongest predictors.

### Step 5: Data Visualization & Evaluation
* Developed a multi-dimensional scatter plot using `seaborn` to visualize **Support vs. Confidence**, with the size and color hue of the data points representing the **Lift** score.
* Plotted the Top 10 strongest rules to visually isolate the "Sure Bets" and "Volume Drivers" for the business.

---

## 📊 Key Business Insights & Strategies
The machine learning model translated raw data into four distinct, data-backed business strategies to hit the 25% growth target:

1. **The "Volume Driver" (Store Layout):** The analysis identified a specific combination occurring in 15% of all store transactions. **Strategy:** Immediately redesign physical store layouts to place these high-volume book categories adjacent to each other.
2. **The "High-Lift" Niche (Online Engine):** The model revealed a hidden relationship with a Lift > 2.30, meaning buying one book more than doubles the likelihood of buying the other. **Strategy:** Implement this exact pairing as the trigger for the online store's "Frequently Bought Together" recommendation engine.
3. **The "Sure Bets" (Targeted Bundling):** Identified pairings with lower overall volume but maximum confidence (>60%). **Strategy:** Bundle these specific books together with a slight discount to easily convert single-book buyers and increase average transaction size.
4. **Inventory Management:** Leveraging these specific correlation metrics allows the bookstore to intelligently anticipate customer needs, optimize physical shelf space constraints, and prevent stockouts of highly dependent categories.

---

## 📂 Repository Files
* `association_rule_assignment.py`: The complete Python script including the database pipeline, model building, and evaluation.
* `book.csv`: The raw transactional dataset (2000 records, 11 categories).

## ⚙️ How to Run
1. Ensure you have MySQL installed and running locally.
2. Update the database credentials (`user`, `pw`, `db`) in the `sqlalchemy` engine string to match your local environment.
3. Install the required dependencies: `pip install pandas matplotlib seaborn mlxtend sqlalchemy pymysql`
4. Run the Python script to view the generated rules and visualizations.
