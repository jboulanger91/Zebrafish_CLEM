Zebrafish Hindbrain Functional Connectomics

Understanding how neuronal structure gives rise to function is a central challenge in neuroscience.
This repository brings together complementary approaches to study the zebrafish hindbrain — a model system for sensory processing and motor control — by integrating high-resolution anatomical data with physiological and computational analyses.

The project combines:
	•	Morphological reconstruction of individual neurons and their projections,
	•	Synaptic connectivity analysis across identified neuronal populations, and
	•	Network-level modeling constrained by experimentally derived connectomes.

Through these efforts, we aim to uncover the structural and computational principles that govern sensorimotor transformations in the hindbrain and link them to behaviorally relevant neural dynamics.

Each component of the repository corresponds to a specific stage in this integrative workflow:
	1.	Uploading and organizing neuronal morphologies and metadata
	2.	Registering data to a standardized zebrafish reference brain
	3.	Performing connectivity analyses and visualization
	4.	Predicting functional neuron types based on morphology
	5.	Simulating circuit dynamics under connectome-based constraints

The code, environments, and documentation are designed to support reproducible and modular collaboration across teams working on different aspects of the connectome-to-function pipeline.

⸻

Repository Structure

1. Uploading Neuronal Morphologies and Metadata

Tools and scripts for preparing, validating, and uploading reconstructed neuronal morphologies along with associated metadata to shared databases or archives.

Folder: Uploading_neuronal_morphologies/
Environment file: env.yaml

⸻

2. Reference Brain Registration

Pipelines for aligning neuronal morphologies, imaging data, and other spatial datasets to a common zebrafish reference brain coordinate framework.

Folder: Reference_brain_registration/
Environment file: env.yaml

⸻

3. Connectivity Analysis

Code for analyzing synaptic connectivity, generating adjacency and weight matrices, and visualizing network organization at different scales.

Folder: Connectivity_analysis/
Environment file: env.yaml

⸻

4. Morphology-Based Prediction of Neuronal Functional Types

Scripts and models for predicting neuronal functional identities from morphological features, such as soma position, dendritic structure, and axonal projection patterns.

Folder: Morphology_based_prediction_of_neuronal_functional_types/
Environment file: env.yaml

⸻

5. Connectome-Constrained Network Modeling

Computational models that simulate or predict neural dynamics using empirically derived connectivity constraints and experimentally validated cell-type rules.

Folder: Connectome_constrained_network_modeling/
Environment file: env.yaml