import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings(
    "ignore",
    message="Use the 'save_best_solutions' parameter with caution*"
)
# ---------------
# IMPORTS
# ---------------

import pandas as pd
import numpy as np
import pygad

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ---------------
# LOAD DATASET
# ---------------

data = pd.read_csv("data/winequality.csv", sep=";")

X = data.drop("quality", axis=1)

y = data["quality"].astype(str).str.strip()
y = y.map({"bad": 0, "good": 1})

feature_names = np.array(X.columns)
num_features = X.shape[1]

X = X.to_numpy()

# 1️⃣ First split: create TEST set (used ONCE at the end)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 2️⃣ Second split: create VALIDATION set (used inside GA)
X_train2, X_val, y_train2, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.25,   # 25% of 80% = 20% total
    random_state=42,
    stratify=y_train
)

# ---------------
# BASELINE MODEL
# ---------------

pipeline = Pipeline([
    ("scaler", StandardScaler()),   # IMPORTANT for KNN
    ("knn", KNeighborsClassifier(
        n_neighbors=5,
        weights="distance"
        ))
    ])

# =====BASELINE MODEL======

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

baseline_acc = accuracy_score(y_test, y_pred)

# =====FITNESS FUNCTION======

def fitness_function(ga_instance, solution, solution_idx):

    if np.sum(solution) == 0:
        return 0

    selected_idx = np.where(solution == 1)[0]

    X_train_subset = X_train2[:, selected_idx]
    X_val_subset   = X_val[:, selected_idx]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(
            n_neighbors=5,
            weights="distance"
        ))
    ])

    pipeline.fit(X_train_subset, y_train2)
    y_pred = pipeline.predict(X_val_subset)

    acc = accuracy_score(y_val, y_pred)
    return acc

def on_generation(ga_instance):
    gen = ga_instance.generations_completed
    best_fit = ga_instance.best_solutions_fitness[-1]

    print(f"Generation {gen} | Best Fitness: {best_fit:.4f}")

# ---------------
# GA CONFIG
# ---------------

ga = pygad.GA(
    num_generations=10,
    sol_per_pop=30,
    num_parents_mating=15,
    fitness_func=fitness_function,
    num_genes=num_features,
    gene_space=[0, 1],
    parent_selection_type="rws",
    crossover_type="uniform",
    mutation_type="random",
    keep_elitism=1,
    mutation_num_genes=1,
    random_seed=42,
    save_best_solutions=True
)

# ---------------
# GA RUN
# ---------------

ga.run()

print("\n" + "=" * 60)
print("GA FITNESS SUMMARY (VALIDATION SET)")
print("=" * 60)

# претвори ги од numpy floats во python floats и заокружи
fitness_list = [float(f) for f in ga.best_solutions_fitness]

for i, f in enumerate(fitness_list):
    print(f"Checkpoint {i:02d}: {f:.4f}")

print("-" * 60)

# од листава најди го [најдобриот] индексот на чие позиција е највисоката точност/fitness 
best_idx = np.argmax(fitness_list) # -> враќа индекс

# најди го решението во листата на решенија (feature subsets) на позицијата на [најдобриот] индекс 
best_solution = ga.best_solutions[best_idx] # -> враќа [1, 1, 0, 0]

# види тука која му е точноста 
best_fitness = fitness_list[best_idx] # -> враќа точност/fitness 0.9786

print(f"GLOBAL best validation fitness: {best_fitness:.4f}")

# -----------------------------
# Selected features
# -----------------------------
# дај ги тие индекси/позиции на колони чија што позиција им е 1
selected_features = feature_names[best_solution == 1] 

print("\n" + "=" * 60)
print("FINAL SELECTED FEATURE SUBSET")
print("=" * 60)
print(f"Selected features: {len(selected_features)} / {len(feature_names)}\n")

for i, feature in enumerate(selected_features, start=1):
    print(f"  {i:02d}. {feature}") # кои се колоните

# -----------------------------
# FINAL TEST EVALUATION
# -----------------------------
print("\n" + "=" * 60)
print("FINAL TEST SET EVALUATION")
print("=" * 60)

idx = np.where(best_solution == 1)[0] # at what positions is the value equal to 1?

X_train_fs = X_train[:, idx] # idx = [2, 4, 1] these are column indices
X_test_fs = X_test[:, idx]

pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(
            n_neighbors=5,
            weights="distance"
        ))
])

pipeline.fit(X_train_fs, y_train)
y_pred = pipeline.predict(X_test_fs) # и на крај со ТЕСТ множество пак ама со СКРАТЕНИ колони

final_acc = accuracy_score(y_test, y_pred)

print(f"- Baseline Accuracy (ALL features): {baseline_acc:.4f}")
print(f"\n- Final test accuracy: {final_acc:.4f}")
print("=" * 60)