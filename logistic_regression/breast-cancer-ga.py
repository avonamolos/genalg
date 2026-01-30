import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings(
    "ignore",
    message="Use the 'save_best_solutions' parameter with caution*"
)

# ------------
# IMPORTS
# ------------

import numpy as np
import pygad

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# ---------------
# LOAD DATASET
# ---------------
data = load_breast_cancer()

X = data.data
y = data.target  # 0 = malignant, 1 = benign

feature_names = data.feature_names
num_features = X.shape[1]

print("Dataset shape:", X.shape)
print("Number of features:", num_features)
print("Class distribution:", np.bincount(y))

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
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

baseline_acc = accuracy_score(y_test, y_pred)

# ------------------
# FITNESS FUNCTION
# ------------------

def fitness_function(ga_instance, solution, solution_idx):

    if np.sum(solution) == 0:
        return 0

    selected_idx = np.where(solution == 1)[0]

    X_train_subset = X_train2[:, selected_idx]
    X_val_subset   = X_val[:, selected_idx]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            random_state=42
        ))
    ])

    pipeline.fit(X_train_subset, y_train2)
    y_pred = pipeline.predict(X_val_subset)

    acc = accuracy_score(y_val, y_pred)
    return acc

# ---------------
# ON GENERATION
# ---------------

# def on_generation(ga_instance):
    
#     gen_number = ga_instance.generations_completed

#     if gen_number % 10 == 0:
#         print(
#             f"Generation {ga_instance.generations_completed} | "
#             f"Best Fitness: {ga_instance.best_solution()[1]:.4f}"
#         )

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
    keep_elitism=1, # исто да стои најдобрата понатаму ама ова понатаму може и да се смени бидејќи со мутации, кросовери
    mutation_num_genes=1,
    random_seed=42,
    save_best_solutions=True # за да ми седи секогаш најдобрата некогаш што е најдена од генерација
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
    ("logreg", LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        random_state=42
    ))
])

pipeline.fit(X_train_fs, y_train)
y_pred = pipeline.predict(X_test_fs) # и на крај со ТЕСТ множество пак ама со СКРАТЕНИ колони

final_acc = accuracy_score(y_test, y_pred)

print(f"- Baseline Accuracy (ALL features): {baseline_acc:.4f}")
print(f"\n- Final test accuracy: {final_acc:.4f}")
print("=" * 60)



# ga.run()

# solution, solution_fitness, _ = ga.best_solution()

# selected_features = feature_names[solution == 1]

# print("\nBest GA solution:")
# print("Selected features:", list(selected_features))
# print("Number of features:", len(selected_features))
# print("Accuracy:", solution_fitness)

