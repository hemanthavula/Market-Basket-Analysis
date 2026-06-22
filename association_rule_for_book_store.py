'''


# Data:
# The book purchase transaction details are obtained to analyze buying behavior.
# The shape is (2000, 11)

# Description:
# A dataset containing binary transactional records (0 = not purchased, 1 = purchased) 
# across 11 different book categories (ChildBks, YouthBks, CookBks, etc.).
'''

# DATA DICTIONARY

# - ChildBks  : Purchase of Children's Books
# - YouthBks  : Purchase of Youth Books
# - CookBks   : Purchase of Cookbooks
# - DoItYBks  : Purchase of Do-It-Yourself Books
# - RefBks    : Purchase of Reference Books
# - ArtBks    : Purchase of Art Books
# - GeogBks   : Purchase of Geography Books
# - ItalCook  : Purchase of Italian Cookbooks
# - ItalAtlas : Purchase of Italian Atlases
# - ItalArt   : Purchase of Italian Art Books
# - Florence  : Purchase of books on Florence


# Import all required libraries 
import pandas as pd  # Importing pandas library and aliasing it as pd for easier reference.
import matplotlib.pyplot as plt  # Importing matplotlib's pyplot module and aliasing it as plt for easier reference.
import seaborn as sns # Importing seaborn for advanced data visualization
from mlxtend.frequent_patterns import apriori, association_rules  # Importing specific functions from mlxtend library.

from sqlalchemy import create_engine  # Importing create_engine function from sqlalchemy library.
from urllib.parse import quote
import pickle  # Importing pickle module for serialization.



# Create a SQLAlchemy engine to connect to the MySQL database
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}".format(user = "root", pw = quote("Hemanth@17"), db = "udb"))

# Reading csv file with help of pandas 
data = pd.read_csv(r"C:\Users\heman\Downloads\Data Set\book.csv")

# display first 5 rows
data.head(5) 

# display last 5 rows
data.tail(5)

# info about data
data.info()

# descriptive statitics of  data
describe = data.describe()

# shape of data
data.shape

# writing data to mysql database
data.to_sql('book_store', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

# reading from sql database 
sql = 'select * from book_store;'
book_store = pd.read_sql_query(sql, con = engine)



# Data Pre-processing
# Convert transactional binary matrix (0s and 1s) to boolean format
book_store = book_store.astype(bool) 

# Model Building: Apriori Algorithm
print("\n Running Apriori Algorithm ")
    
# Generate Frequent Itemsets (min support 10%)
frequent_itemsets = apriori(book_store, min_support=0.1, use_colnames=True)
print(f"Found {len(frequent_itemsets)} frequent itemsets.")

# Generate Association Rules (Metric: Lift, Threshold: > 1.2)
print("\n[INFO] Generating Association Rules...")
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)

# Sort rules by highest lift
rules = rules.sort_values('lift', ascending=False)
    

# Create a new column that combines Antecedents and Consequents into a single set
# Because a set is unordered, {ChildBks, CookBks} will match {CookBks, ChildBks}
rules['combined_items'] = rules.apply(lambda row: frozenset(row['antecedents'] | row['consequents']), axis=1)

# Drop duplicates based on this new combined column, keeping the first occurrence (highest lift)
rules_no_duplicates = rules.drop_duplicates(subset=['combined_items'], keep='first')

# Drop the temporary 'combined_items' column to clean up the final dataframe
rules_no_duplicates = rules_no_duplicates.drop(columns=['combined_items'])

print(f"Original number of rules: {len(rules)}")
print(f"Number of rules after removing reversed duplicates: {len(rules_no_duplicates)}")

# Display the cleaned rules
print("\n--- Unique Association Rules ---")
print(rules_no_duplicates[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))


# Model Evaluation / Visualization
print("\n Generating Visualizations ")
    
plt.figure(figsize=(10, 6))

# Scatter plot of Support vs Confidence
sns.scatterplot(
        x="support", 
        y="confidence", 
        size="lift", 
        hue="lift", 
        sizes=(50, 250), 
        data=rules_no_duplicates, 
        palette="viridis",
        alpha=0.8
    )

plt.title('Association Rules: Support vs. Confidence (Colored by Lift)')


top10_rules = rules_no_duplicates.head(10)


# Plot a scatter plot for the top 10 rules, where support and confidence are represented by x and y axes respectively, and lift is represented by color
top10_rules.plot(x = "support", y = "confidence", c = top10_rules.lift, 
             kind = "scatter", s = 30, cmap = plt.cm.coolwarm)


''' High-confidence pairings identified in the analysis are the strongest, most reliable predictors of future book purchases.

Every top rule exhibits a Lift score over 2.0, proving these aren't random coincidences but highly significant buying patterns.

Rules with low support but maximum confidence (>60%) are your "Sure Bets," making them ideal candidates for targeted discount bundling.

By bundling these highly correlated "Sure Bet" books, you can easily convert the remaining single-book buyers to increase transaction sizes.

The "Volume Driver" rule, occurring in 15% of all store transactions, represents your most reliable, high-traffic book combination.

You should immediately redesign the physical store layout to place these high-volume book categories directly next to each other.

The rule with the maximum lift (>2.30) reveals a hidden relationship where buying one book more than doubles the chance of buying the other.

This high-lift niche rule is the perfect trigger to implement in your online store's "Frequently Bought Together" recommendation engine.

Leveraging these rules allows Kitabi Duniya to intelligently anticipate customer needs, prevent stockouts, and manage inventory effectively.

Executing these specific, data-backed cross-selling strategies provides a direct and efficient roadmap to achieving your 25% growth objective.'''
