import streamlit as st
import matplotlib.pyplot as plt
import sys
import os

# Direct import
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
            ["Upload Custom Dataset", "Use Sample Dataset"]
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
            self.filepath = "path/to/your/default/dataset.csv"

    def render_ui(self):
        st.title("Multi-Objective Genetic Algorithm for Artificial Neural Network Hyperparameter Optimization")

        # Dataset Loading
        self.load_dataset()

        # Sidebar for Configuration
        st.sidebar.header("Genetic Algorithm Parameters")

        # Hyperparameter Sliders
        population_size = st.sidebar.slider(
            "Population Size",
            min_value=1,
            max_value=100,
            value=20,
            help="Number of individuals in each generation"
        )

        max_generations = st.sidebar.slider(
            "Max Generations",
            min_value=1,
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
        if st.button("Run Multi-Objective Genetic Algorithm Optimization"):
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
            with st.spinner('Running Multi-Objective Genetic Algorithm Optimization \n This may take some time.....'):
                self.optimizer.run_optimization()

            # Display Results
            self.display_results()

    def display_results(self):
        st.subheader("Optimization Results")

        # Create two columns for side-by-side plots
        col1, col2 = st.columns(2)

        # Accuracy Convergence in first column
        with col1:
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            ax1.plot(self.optimizer.generation_best_accuracy, marker='o')
            ax1.set_title("Best Accuracy per Generation")
            ax1.set_xlabel("Generation")
            ax1.set_ylabel("Accuracy")
            st.pyplot(fig1)

        # Loss Convergence in second column
        with col2:
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            ax2.plot(self.optimizer.generation_best_loss, marker='o', color='red')
            ax2.set_title("Best Loss per Generation")
            ax2.set_xlabel("Generation")
            ax2.set_ylabel("Loss")
            st.pyplot(fig2)

        # Best Individual Details below the plots
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


def main():
    ui = GeneticAlgorithmUI()
    ui.render_ui()


if __name__ == "__main__":
    main()