import warnings
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all="ignore")
warnings.filterwarnings(
    "ignore",
    message="Use the 'save_best_solutions' parameter with caution*"
)

import pygad
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X = np.loadtxt("data/madelon/MADELON/madelon_train.data")
y = np.loadtxt("data/madelon/MADELON/madelon_train.labels")

# convert labels {-1, +1} → {0, 1}
y = (y == 1).astype(int)

print(X.shape)  # (2000, 500)
print(np.unique(y))


X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.3,      # 70% train, 30% validation
    random_state=42,
    stratify=y
)

baseline_model = Pipeline([
    ("scaler", StandardScaler()),   # IMPORTANT for KNN
    ("knn", KNeighborsClassifier(
        n_neighbors=5,
        weights="distance"
        ))
    ])

baseline_model.fit(X_train, y_train)
y_pred = baseline_model.predict(X_val)

baseline_acc = accuracy_score(y_val, y_pred)
print("Baseline accuracy (ALL 500 features):", baseline_acc)


def fitness_function(ga_instance, solution, solution_idx):

    num_selected = np.sum(solution)
    if num_selected == 0:
        return 0

    selected_idx = np.where(solution == 1)[0]

    X_train_subset = X_train[:, selected_idx]
    X_val_subset   = X_val[:, selected_idx]

    model = Pipeline([
    ("scaler", StandardScaler()),   # IMPORTANT for KNN
    ("knn", KNeighborsClassifier(
        n_neighbors=5,
        weights="distance"
        ))
    ])

    model.fit(X_train_subset, y_train)
    acc = accuracy_score(y_val, model.predict(X_val_subset))

    # 🔑 penalty term
    penalty = 0.0008 * num_selected   # tuneable
    return acc - penalty

def on_generation(ga_instance):
    gen = ga_instance.generations_completed
    best_fit = ga_instance.best_solutions_fitness[-1]

    if gen < 10:
        print(f"Generation {gen} | Best Fitness: {best_fit:.4f}")

    elif gen >= 10 and gen % 10 == 0:
        print(f"Generation {gen} | Best Fitness: {best_fit:.4f}")

ga = pygad.GA(
    num_generations=50,
    sol_per_pop=60,
    num_parents_mating=30,
    fitness_func=fitness_function,
    num_genes=500,
    gene_space=[0, 1],
    parent_selection_type="rws",
    crossover_type="uniform",
    mutation_type="random",
    mutation_num_genes=10,
    keep_elitism=2,
    random_seed=42,
    on_generation=on_generation,
    save_best_solutions=True
)

ga.run()

best_idx = np.argmax(ga.best_solutions_fitness)

best_solution = ga.best_solutions[best_idx]
best_fitness = ga.best_solutions_fitness[best_idx]

selected_idx = np.where(best_solution == 1)[0]
num_selected = len(selected_idx)

X_train_fs = X_train[:, selected_idx]
X_val_fs   = X_val[:, selected_idx]

final_model = Pipeline([
    ("scaler", StandardScaler()),   # IMPORTANT for KNN
    ("knn", KNeighborsClassifier(
        n_neighbors=5,
        weights="distance"
        ))
    ])

final_model.fit(X_train_fs, y_train)
final_val_acc = accuracy_score(y_val, final_model.predict(X_val_fs))

print("\n" + "="*50)
print("MADELON DATASET")
print("KNN + Genetic Algorithm")
print("="*50)

print(f"\nDataset shape           : {X.shape}")
print(f"Class labels            : {np.unique(y)}")

print("\n" + "-"*50)
print("BASELINE (NO FEATURE SELECTION)")
print("-"*50)
print(f"Validation accuracy (500 features): {baseline_acc:.4f}")

print("\n" + "-"*50)
print("GA FEATURE SELECTION RESULTS")
print("-"*50)
print(f"Best GA fitness (penalized) : {best_fitness:.4f}")
print(f"Selected features          : {num_selected}")
print(f"Dimensionality reduction   : {X.shape[1]} → {num_selected}")

print("\n" + "-"*50)
print("FINAL MODEL (GA FEATURES)")
print("-"*50)
print(f"Validation accuracy        : {final_val_acc:.4f}")

print("\n" + "="*50)


# print("\nGA convergence (best fitness per generation):")
# for i, f in enumerate(ga.best_solutions_fitness, start=1):
#     print(f"Gen {i:02d}: {f:.4f}")


