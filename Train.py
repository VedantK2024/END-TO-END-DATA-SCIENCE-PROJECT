"""
Model Training and Evaluation
"""
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Load preprocessed data
X_train = np.load('data/X_train.npy')
X_test = np.load('data/X_test.npy')
y_train = np.load('data/y_train.npy')
y_test = np.load('data/y_test.npy')

print("=" * 60)
print("MODEL TRAINING AND EVALUATION")
print("=" * 60)

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=10),
    'Lasso Regression': Lasso(alpha=10),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
}

results = {}

print("\nTraining and evaluating models...\n")

for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    results[name] = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'model': model
    }
    
    print(f"{name}:")
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Test RMSE: ${test_rmse:,.2f}")
    print(f"  Test MAE: ${test_mae:,.2f}")
    print()

# Find best model
best_model_name = max(results, key=lambda x: results[x]['test_r2'])
best_model = results[best_model_name]['model']

print("=" * 60)
print(f"BEST MODEL: {best_model_name}")
print(f"Test R² Score: {results[best_model_name]['test_r2']:.4f}")
print("=" * 60)

# Save best model
with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print(f"\nBest model saved to models/best_model.pkl")

# Create comparison visualization
plt.figure(figsize=(12, 5))

# R² scores comparison
plt.subplot(1, 2, 1)
model_names = list(results.keys())
test_r2_scores = [results[name]['test_r2'] for name in model_names]
plt.barh(model_names, test_r2_scores, color='steelblue')
plt.xlabel('R² Score')
plt.title('Model Performance Comparison (Test R²)')
plt.xlim(0, 1)

# Actual vs Predicted for best model
plt.subplot(1, 2, 2)
y_pred_best = best_model.predict(X_test)
plt.scatter(y_test, y_pred_best, alpha=0.6, edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title(f'Actual vs Predicted ({best_model_name})')

plt.tight_layout()
plt.savefig('static/model_evaluation.png', dpi=300, bbox_inches='tight')
print("Model evaluation plot saved to static/model_evaluation.png")

# Save results
with open('models/training_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\nTraining complete!")
