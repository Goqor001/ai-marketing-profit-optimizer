import pandas as pd
import numpy as np
import random
random.seed(42)

def clean_data(df):
    df=df.copy()
    df["age"] = pd.to_numeric(df["age"],errors="coerce")
    df["income"] = pd.to_numeric(df["income"],errors="coerce")
    df["credit_score"] = pd.to_numeric(df["credit_score"],errors="coerce")
    df["experience"] = pd.to_numeric(df["experience"],errors="coerce")
    df["website_visits"] = pd.to_numeric(df["website_visits"],errors="coerce")
    df["time_on_site"] = pd.to_numeric(df["time_on_site"],errors="coerce")

    return df

def get_will_buy(row):
    score = 0
    engagement = row["website_visits"]*row["time_on_site"]

    if row["income"] > 900:
        score += 1
    if engagement > 100:
        score += 1
    if row["experience"] > 5:
        score += 1
    if row["credit_score"] > 600:
        score += 1
    elif row["credit_score"] <= 600:
        score -= 1 

    if score == 3:
        prob = 0.8
    elif score == 2:
        prob = 0.6
    elif score == 1:
        prob = 0.4
    else:
        prob = 0.2
    
    rand = random.random()
    if prob > rand:
        return True
    else:
        return False

def find_best_strategy(x_test_log, x_test_tree):

    best_profit_log = float("-inf")
    best_profit_tree = float("-inf")

    best_budget_log = 0
    best_budget_tree = 0

    best_cost_log = 0
    best_cost_tree = 0

    for budget in range(1, 31):
        for cost in [10, 20, 30, 40]:

            top_log = x_test_log.head(budget).copy()
            top_log["value"] = top_log["income"] * 0.1
            buyers_log = top_log[top_log["real"] == True]

            revenue_log = buyers_log["value"].sum()
            cost_log = len(top_log) * cost
            profit_log = revenue_log - cost_log

            top_tree = x_test_tree.head(budget).copy()
            top_tree["value"] = top_tree["income"] * 0.1
            buyers_tree = top_tree[top_tree["real"] == True]

            revenue_tree = buyers_tree["value"].sum()
            cost_tree = len(top_tree) * cost
            profit_tree = revenue_tree - cost_tree

            if profit_log > best_profit_log:
                best_profit_log = profit_log
                best_budget_log = budget
                best_cost_log = cost

            if profit_tree > best_profit_tree:
                best_profit_tree = profit_tree
                best_budget_tree = budget
                best_cost_tree = cost
        
 

    if best_profit_tree > best_profit_log:
        print("\n---DEBUG---")
        print("Best Log:", best_budget_log,best_cost_log,best_profit_log)
        print("Best Tree:",best_budget_tree,best_cost_tree,best_profit_tree)
        return {
            "model": "tree",
            "budget": best_budget_tree,
            "cost": best_cost_tree,
            "profit": best_profit_tree
        }
    else:
        print("\n---DEBUG---")
        print("Best Log:", best_budget_log,best_cost_log,best_profit_log)
        print("Best Tree:",best_budget_tree,best_cost_tree,best_profit_tree)
        return {
            "model": "logistic",
            "budget": best_budget_log,
            "cost": best_cost_log,
            "profit": best_profit_log
        }

best_profit_threshold_log = float("-inf")
best_threshold_log = 0
best_profit_threshold_tree = float("-inf")
best_threshold_tree = 0

def train_models(x_train,y_train,x_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    log = LogisticRegression(max_iter=5000)
    log.fit(x_train,y_train)

    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(x_train,y_train)
    
    return log, tree

def run_pipeline(file_path):
    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError("DataFrame is empty")
    required_cols = ["age","income","credit_score","city","experience","time_on_site","website_visits"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {','.join(missing)}")
    
    df = clean_data(df)
    df["will_buy"] = df.apply(get_will_buy,axis=1)
    x = df[["age","income","credit_score","city","experience","time_on_site","website_visits"]]
    y = df["will_buy"]
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)
    x_train_copy = x_train.copy()
    x_train_copy["will_buy"] = y_train
    x_train["time_per_visits"] = x_train["time_on_site"]/(x_train["website_visits"]+1)
    x_test["time_per_visits"] = x_test["time_on_site"]/(x_test["website_visits"]+1) 
    x_train["income_per_exp"] = x_train["income"]/(x_train["experience"]+1)
    x_test["income_per_exp"] = x_test["income"]/(x_test["experience"]+1)
    x_train["engagement"] = x_train["website_visits"] * x_train["time_on_site"]
    x_test["engagement"] = x_test["website_visits"] * x_test["time_on_site"]

    global_mean = x_train_copy["will_buy"].mean()
    city_stats = x_train_copy.groupby("city")["will_buy"].agg(["mean","count"])
    k = 5
    city_stats["smooth"] = (
       (city_stats["mean"]*city_stats["count"]+global_mean*k)
       / (city_stats["count"]+k)
    )
    x_train["city_score"] = x_train["city"].map(city_stats["smooth"])
    x_test["city_score"] = x_test["city"].map(city_stats["smooth"])
    x_train["high_income"] = (x_train["income"]>800).astype(int)
    x_test["high_income"] = (x_test["income"]>800).astype(int)



    features = [
       "income",
       "experience",
       "time_on_site",
       "website_visits",
       "engagement",
       "income_per_exp",
       "high_income"
    ]

    x_train = x_train[features]
    x_test = x_test[features]
    
    log, tree = train_models(x_train,y_train,x_test)
    log_prob = log.predict_proba(x_test)
    tree_prob = tree.predict_proba(x_test)

    x_test_log = x_test.copy()

    x_test_log["probability"] = log_prob[:, 1]
    x_test_log["real"] = y_test.values
    x_test_log = x_test_log.sort_values(by="probability", ascending=False)

    x_test_tree = x_test.copy()
    x_test_tree["probability"] = tree_prob[:, 1]
    x_test_tree["real"] = y_test.values
    x_test_tree = x_test_tree.sort_values(by="probability", ascending=False)

    from sklearn.metrics import accuracy_score
    log_acc = accuracy_score(y_test,log.predict(x_test))
    tree_acc = accuracy_score(y_test,tree.predict(x_test))

    print("\nMODEL METRICS:")
    print("LogisticRegression accuracy: ",log_acc)
    print("DecisionTree accuracy: ",tree_acc)

    strategy = find_best_strategy(x_test_log,x_test_tree)
    return {
        "strategy": strategy,
        "log_accuracy": log_acc,
        "tree_accuracy": tree_acc
    }

result = run_pipeline(r"C:\Users\hovha\OneDrive\Рабочий стол\ai-marketing-profit-optimizer\marketing_ml.csv")

print("\nFINAL STRATEGY:")
print("Model:", result["strategy"]["model"])
print("Budget:", result["strategy"]["budget"])
print("Cost per client:", result["strategy"]["cost"])
print("Expected profit:", result["strategy"]["profit"])

print("\nMODEL METRICS:")
print("LogisticRegression accuracy:", result["log_accuracy"])
print("DecisionTree accuracy:", result["tree_accuracy"])

print("\nCOMPARISON:")
if result["strategy"]["model"] == "tree":
    print("Chosen model: Decision Tree (better for profit)")
    print("But Logistic Regression may be more stable")
else:
    print("Chosen model: Logistic Regression (more stable)")
    print("Tree might overfit but sometimes gives higher profit")

# best_log_profit = float("-inf")
# best_tree_profit = float("-inf")
# for percent in np.arange(0.1,1.0,0.1):
#     k = int(len(x_test_log)*percent)
#     top_k = x_test_log.head(k)
#     top_k["value"] = top_k["income"]*0.1
#     buyers_log = top_k[top_k["real"]==True]
#     revenue_log = buyers_log["value"].sum()
#     cost_log = len(top_k) * 20 
#     profit_log = revenue_log - cost_log

#     top_p = x_test_tree.head(k)
#     top_p["value"] = top_p["income"]*0.1
#     buyers_tree = top_p[top_p["real"]==True]
#     revenue_tree = buyers_tree["value"].sum()
#     cost_tree = len(top_p) * 20 
#     profit_tree = revenue_tree - cost_tree

#     if profit_log > best_log_profit:
#         best_log_profit = profit_log
#         best_precent_k = percent
#     if profit_tree > best_tree_profit:
#         best_tree_profit = profit_tree
#         best_precent_p = percent

# print("Log` top%: ",best_precent_k,"Best_profit: ",best_log_profit)
# print("Tree` top%: ",best_precent_p,"Best_profit: ",best_tree_profit)

#  for threshold in np.arange(0.1,1.0,0.1):
    # x_test_log_copy = x_test_log.copy()
    # x_test_log_copy = x_test_log_copy[x_test_log_copy["probability"]>threshold]
    # threshold_profit_log = (x_test_log_copy["real"].sum() * 70)-(len(x_test_log_copy)*20)

    # if threshold_profit_log > best_profit_threshold_log:
    #     best_profit_threshold_log = threshold_profit_log
    #     best_threshold_log = threshold
    
    # x_test_tree_copy = x_test_tree.copy()
    # x_test_tree_copy = x_test_tree_copy[x_test_tree_copy["probability"]>threshold]
    # threshold_profit_tree = (x_test_tree_copy["real"].sum() * 70)-(len(x_test_tree_copy)*20)

    # if threshold_profit_tree > best_profit_threshold_tree:
    #     best_profit_threshold_tree = threshold_profit_tree
    #     best_threshold_tree = threshold
    

# print("Log threshold` best_threshold: ",best_threshold_log,"best_profit: ",best_profit_threshold_log)
# print("Tree threshold` best_threshold: ",best_threshold_tree,"best_profit: ",best_profit_threshold_tree)

# from sklearn.metrics import accuracy_score
# print("accuracy_score: ",accuracy_score(y_test,log.predict(x_test)))
# print(accuracy_score(y_train,log.predict(x_train)))
# print(pd.Series(tree.feature_importances_,index=x_train.columns))