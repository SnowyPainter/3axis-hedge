import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def features_MDA_importances(X_test, y_test, model, original_accuracy, features):
    """
    Calculate feature importance using Mean Decrease Accuracy (MDA) method.
    
    Parameters:
    - X_test: np.ndarray, Test features
    - y_test: np.ndarray, Test labels
    - model: Trained model with a predict method
    - original_accuracy: float, Accuracy of the model on the original test set
    - features: List[str], Feature names corresponding to X_test columns
    
    Returns:
    - feature_importance_df: pd.DataFrame, Feature importances sorted by importance
    """
    feature_importance = []
    X_test_copy = X_test.copy()
    
    for i, feature in enumerate(features):
        # 특성 열을 무작위로 섞음
        shuffled_column = np.random.permutation(X_test[:, i])
        X_test_copy[:, i] = shuffled_column
        
        # 모델 예측
        y_pred_shuffled = model.predict(X_test_copy)
        shuffled_accuracy = accuracy_score(y_test, y_pred_shuffled)
        
        # 중요도 계산 (성능 저하)
        importance = original_accuracy - shuffled_accuracy
        feature_importance.append({
            'feature': feature,
            'importance': importance
        })
        
        # 원래 데이터 복원
        X_test_copy[:, i] = X_test[:, i]
    
    # 중요도 정렬
    feature_importance_df = pd.DataFrame(feature_importance).sort_values(by='importance', ascending=False)
    
    # 출력
    print("\nFeature Importance (MDA):")
    print(feature_importance_df)
    
    return feature_importance_df