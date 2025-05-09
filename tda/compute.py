# Import necessary GUDHI library components
import gudhi as gd
import numpy as np
import matplotlib.pyplot as plt
import warnings # To suppress potential deprecation warnings

def compute_persistence(points, simplex="rips", max_edge_length=1.5, max_dimension=2, landmarks=None, witnesses=None,
                        **kwargs):
    if simplex == "rips":
        rips_complex = gd.RipsComplex(points=points, max_edge_length=max_edge_length)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
    elif simplex == "witness":
        if landmarks is None or witnesses is None:
            raise ValueError("Landmarks and witnesses must be provided for simplex='witness'")
        print(f"Building Witness complex with {landmarks.shape[0]} landmarks and {witnesses.shape[0]} witnesses...")
        complex_obj = gd.WitnessComplex(landmarks=landmarks, witnesses=witnesses)
        # Pass additional kwargs like filtration_max if needed
        simplex_tree = complex_obj.create_simplex_tree(max_dimension=max_dimension, **kwargs)
    else:
        raise ValueError("Unsupported simplex type. Use 'rips' or 'witness'.")

    # print("Computing persistence...")
    persistence = simplex_tree.persistence()
    # print("Finished computing persistence.")
    return persistence, simplex_tree


# Function to compute Betti curves, handling different GUDHI versions
def compute_betti_curves(simplex_tree, persistence, thresholds):
    betti_curves_data = None
    try:
        betti_curves_data = simplex_tree.betti_curve(thresholds)
    except (AttributeError, TypeError) as e:
        try:
            warnings.warn("Falling back to manual Betti computation from persistence intervals.", UserWarning)
            raise NotImplementedError("Fallback to manual calculation needed.")
        except Exception as e2:
            max_dim_persisted = 0
            if persistence:  # Check if persistence data exists
                dims_present = [dim for dim, _ in persistence]
                if dims_present:
                    max_dim_persisted = max(dims_present)

            betti_curves_list = []
            for t in thresholds:
                betti_at_t = np.zeros(max_dim_persisted + 1, dtype=int)
                for dim, (birth, death) in persistence:
                    if birth <= t and (death > t or np.isinf(death)):
                        if dim <= max_dim_persisted:
                            betti_at_t[dim] += 1
                betti_curves_list.append(betti_at_t)

            if betti_curves_list:
                betti_curves_data = np.array(betti_curves_list)
            else:
                # Handle case where persistence data was empty
                betti_curves_data = np.zeros((len(thresholds), 1))  # Default to dim 0 if no persistence
    return betti_curves_data


# Function to plot the persistence diagram
def plot_persistence_diagram(persistence, title="Persistence Diagram"):
    gd.plot_persistence_diagram(persistence)
    plt.title(title)
    plt.show()


# Function to plot Betti curves
def plot_betti_curves(thresholds, betti_curves_data, title="Persistence Diagram"):
    if betti_curves_data is None or betti_curves_data.size == 0:
        print("No Betti curve data to plot.")
        return

    num_dims = betti_curves_data.shape[1]
    for dim in range(num_dims):
        # Check if the curve for this dimension actually exists (is not all zero)
        if np.any(betti_curves_data[:, dim]):
            plt.plot(thresholds, betti_curves_data[:, dim], label=f'Betti {dim}')

    if plt.gca().has_data():
        plt.xlabel("Filtration Scale (epsilon)")
        plt.ylabel("Number of Features (Betti Number)")
        plt.ylim(bottom=-0.5, top=max(5, np.max(betti_curves_data[:, 1:].max(initial=0)) + 1))
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("No non-zero Betti curves found to plot.")


def analyze_and_plot_tda(embeddings, label, complex="rips", max_filt_scale=1.5, max_hom_dim=2, threshold_count=100,
                         percent_of_landmarks=0.1, plot=True):
    # print(f"\n--- Analyzing {label} Embeddings ({embeddings.shape[0]} samples) ---")
    landmarks, witnesses = None, None
    if complex == "witness":
        witnesses = embeddings
        print(f"  Set witnesses: {witnesses.shape[0]} points.")

        actual_num_landmarks = embeddings.shape[0] * percent_of_landmarks
        landmark_indices = np.random.choice(embeddings.shape[0], int(actual_num_landmarks), replace=False)
        landmarks = embeddings[landmark_indices, :]
        print(f"  Selected landmarks: {landmarks.shape[0]} points (randomly).")
    persistence_intervals, st = compute_persistence(embeddings,
                                                    complex=complex,
                                                    max_edge_length=max_filt_scale,
                                                    max_dimension=max_hom_dim,
                                                    landmarks=landmarks,
                                                    witnesses=witnesses)
    # Note: uncomment this to see the exact persistence intervals
    # if persistence_intervals:
    #     print("\n--- Persistence Intervals for Dim 3 (Unsafe Text) ---")
    #     dim3_intervals = [(b, d) for dim, (b, d) in persistence_intervals if dim == 3]
    #     if dim3_intervals:
    #         print(f"Found {len(dim3_intervals)} Betti 3 features:")
    #         for birth, death in dim3_intervals:
    #             death_str = f"{death:.4f}" if not np.isinf(death) else "inf"
    #             print(f"  [{birth:.4f}, {death_str})")
    #     else:
    #         print("No Betti 3 features found in persistence intervals.")
    threshold_values = np.linspace(0, max_filt_scale, threshold_count)
    betti_data = compute_betti_curves(st, persistence_intervals, threshold_values)

    # print(f"\nPlotting {label} results...")
    if plot:
        plot_persistence_diagram(persistence_intervals, title=f"Persistence Diagram ({label})")
        plot_betti_curves(threshold_values, betti_data, title=f"Betti Curve ({label})")
    return persistence_intervals, betti_data, threshold_values
