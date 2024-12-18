import streamlit as st
import matplotlib.pyplot as plt
import sys
import os

# Add the directory containing your original notebook to the Python path
sys.path.append(os.path.abspath('.'))

# Import your StrokeModelOptimizer
from final_moga_ui import StrokeModelOptimizer


class GeneticAlgorithmUI:
    def __init__(self):
        self.optimizer = None
        self.filepath = None  # Path to your dataset

    def load_dataset(self):
        """Allow user to upload or select a dataset"""
        st.sidebar.header("Dataset Selection")
        dataset_option = st.sidebar.selectbox(
            "Choose Dataset Source",
            ["Use Sample Dataset", "Upload Custom Dataset"]
        )

        if dataset_option == "Upload Custom Dataset":
            uploaded_file = st.sidebar.file_uploader(
                "Choose a CSV file",
                type="csv"
            )
            if uploaded_file is not None:
                # Save uploaded file temporarily
                self.filepath = f"uploaded_dataset_{uploaded_file.name}"
                with open(self.filepath, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.sidebar.success("File uploaded successfully!")
        else:
            # Use a default dataset path
            self.filepath = "/content/sample_data/cleaned-stroke-prediction-dataset-balanced.csv"

    def render_ui(self):
        st.title("Genetic Algorithm for ANN Hyperparameter Optimization")

        # Dataset Loading
        self.load_dataset()

        # Sidebar for Configuration
        st.sidebar.header("Genetic Algorithm Parameters")

        # Hyperparameter Sliders
        population_size = st.sidebar.slider(
            "Population Size",
            min_value=10,
            max_value=100,
            value=20,
            help="Number of individuals in each generation"
        )

        max_generations = st.sidebar.slider(
            "Max Generations",
            min_value=5,
            max_value=50,
            value=10,
            help="Total number of generations to evolve"
        )

        crossover_prob = st.sidebar.slider(
            "Crossover Probability",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.1,
            help="Probability of crossover between two parents"
        )

        mutation_prob = st.sidebar.slider(
            "Mutation Probability",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.1,
            help="Probability of mutation in an individual"
        )

        # Run Optimization Button
        if st.button("Run Genetic Algorithm Optimization"):
            # Validate dataset is loaded
            if not self.filepath:
                st.error("Please select or upload a dataset first!")
                return

            # Create optimizer with user-defined parameters
            self.optimizer = StrokeModelOptimizer(self.filepath)

            # Update GA parameters
            self.optimizer.POPULATION_SIZE = population_size
            self.optimizer.MAX_GENERATIONS = max_generations
            self.optimizer.CROSSOVER_PROB = crossover_prob
            self.optimizer.MUTATION_PROB = mutation_prob

            # Run optimization with progress tracking
            with st.spinner('Running Genetic Algorithm Optimization...'):
                self.optimizer.run_optimization()

            # Display Results
            self.display_results()

    def display_results(self):
        # Accuracy Convergence
        st.subheader("Accuracy Convergence")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.optimizer.generation_best_accuracy, marker='o')
        ax.set_title("Best Accuracy per Generation")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Accuracy")
        st.pyplot(fig)

        # Loss Convergence
        st.subheader("Loss Convergence")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.optimizer.generation_best_loss, marker='o', color='red')
        ax.set_title("Best Loss per Generation")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Loss")
        st.pyplot(fig)

        # Best Individual Details
        st.subheader("Best Individual Hyperparameters")
        best_individual = self.optimizer.best_individual
        hyperparameters = {
            "Layer 1 Neurons": int(best_individual[0]),
            "Layer 2 Neurons": int(best_individual[1]),
            "Dropout Rate 1": round(best_individual[2], 4),
            "Dropout Rate 2": round(best_individual[3], 4),
            "Learning Rate": round(best_individual[4], 6),
            "L2 Regularization": round(best_individual[5], 6)
        }
        st.json(hyperparameters)

        # Performance Metrics
        st.subheader("Final Model Performance")
        st.write("These metrics represent the performance of the best model found:")
        st.write("Note: Detailed metrics are printed in the console output")


def main():
    ui = GeneticAlgorithmUI()
    ui.render_ui()


if __name__ == "__main__":
    main()