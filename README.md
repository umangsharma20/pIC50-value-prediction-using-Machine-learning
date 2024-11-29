Ligand Design and Development for Alzheimer's and Parkinson's Disease Targeting CDK5 Using Machine Learning
This project aims to design and develop potential inhibitors for CDK5 (Cyclin-dependent kinase 5), a key enzyme implicated in Alzheimer's and Parkinson's diseases. By leveraging machine learning, we identify compounds that can potentially inhibit CDK5 and predict their bioactivity. The key steps of the process are as follows:

1. SMILES Generation for CDK5
Goal: Identify inhibitors for CDK5 and represent their molecular structures in SMILES format.
Approach:

Search ChEMBL for CDK5-specific molecules.
Filter for human-specific targets and compounds with available IC50 data.
Screen for potential inhibitors and structure the data for further processing.
Save the processed information for descriptor calculations and model building.
2. Molecular Descriptor Calculation for CDK5
Goal: Generate a wide range of molecular descriptors to capture structural and physicochemical properties.
Approach:

Convert SMILES into Isomeric SMILES for standardization.
Compute both 2D and 3D molecular descriptors using specialized computational tools.
Clean the dataset by removing duplicates and irrelevant columns.
Store the calculated descriptors in a structured format suitable for machine learning applications.
3. Fingerprint Generation for CDK5
Goal: Create molecular fingerprints to capture essential structural features.
Approach:

Generate molecular fingerprints, such as ECFP (Extended Connectivity Fingerprints) or FCFP (Functional Class Fingerprints), for the identified CDK5 inhibitors.
Ensure that the fingerprints are in a suitable format for use in machine learning models.
Save the fingerprint data for further analysis.
4. Machine Learning Model for CDK5
Goal: Build a regression model to predict the bioactivity (pIC50) of CDK5 inhibitors.
Approach:

Preprocess the dataset by addressing missing values and filtering molecules based on drug-likeness criteria (e.g., Lipinski’s Rule of Five).
Perform feature selection to identify the most relevant molecular descriptors.
Split the dataset into training (80%) and testing (20%) sets, normalizing input features.
Train a regression model (e.g., Random Forest Regressor) with hyperparameter optimization.
Evaluate the model using performance metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²).
Visualize model predictions and residuals to assess model accuracy and reliability.
5. Model Evaluation
Performance Metrics:

R²: 0.6588
RMSE (Root Mean Squared Error): 0.6935
MSE (Mean Squared Error): 0.4809
MAE (Mean Absolute Error): 0.4845
6. Suggestions for Future Work
Improved Data Representation: Integrate hybrid descriptors that combine both 2D and 3D molecular features for a richer molecular representation.
Alternative Models: Explore advanced algorithms like Gradient Boosting or Neural Networks to improve predictive accuracy.
Validation: Conduct biological validation through docking studies or molecular simulations to confirm the activity of top-ranked inhibitors.
