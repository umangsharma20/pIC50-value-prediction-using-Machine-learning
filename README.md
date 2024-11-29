Design and Development of Ligands for Alzheimer's Disease and Parkinson's disease targeting CDK5 Using Machine Learning1. SMILES Generation for CDK5
Goal: Identify inhibitors for CDK5 and represent their molecular structures in SMILES format.
Approach:
Use ChEMBL resources to search for CDK5-specific molecules.
Apply filters to focus on human-specific targets and select compounds with available IC50 data.
Screen for inhibitors and structure the data for further processing.
Save the processed information for descriptor calculations and model building.
2. Molecular Descriptor Calculation for CDK5
Goal: Generate a wide range of molecular descriptors to represent structural and physicochemical properties.
Approach:
Convert SMILES into Isomeric SMILES for standardization.
Compute descriptors, including both 2D and 3D features, using specialized computational tools.
Preprocess the data by removing duplicates and unnecessary columns.
Save the calculated descriptors in a structured format for machine learning applications.
3. Fingerprint Generation for CDK5
Goal: Create molecular fingerprints to capture essential structural features.
Approach:
Generate molecular fingerprints (e.g., ECFP or FCFP) for CDK5 inhibitors.
Ensure the fingerprints are in a suitable format for machine learning input.
Save the processed fingerprint data for further analysis.
4. Machine Learning Model for CDK5
Goal: Build a regression model to predict the bioactivity (pIC50) of CDK5 inhibitors.
Approach:
Prepare the dataset by addressing missing values and filtering molecules based on drug-likeness criteria such as Lipinski’s rule.
Perform feature selection to identify the most relevant molecular descriptors.
Split the data into training and testing sets (80:20) and normalize input features.
Train regression models, such as Random Forest Regressor, with hyperparameter optimization.
Evaluate the model using metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²).
Visualize predictions and residuals to assess the model's accuracy and reliability.
5. Model Evaluation
Performance Metrics:
R²: 0.6588
RMSE: 0.6935
MSE: 0.4809
MAE: 0.4845
Suggestions for Future Work
Improved Data Representation:
Integrate hybrid descriptors combining both 2D and 3D features for a richer molecular representation.
Alternative Models:
Explore advanced algorithms like Gradient Boosting or Neural Networks to enhance predictive accuracy.
Validation:
Conduct biological validation through docking studies or simulations to confirm the activity of top-ranked molecules.
