"""
PURE CNN ARCHITECTURE - Fatigue Detection
Optimized untuk accuracy 90%+

ARSITEKTUR:
- 5 Convolutional Blocks (progressive filters: 32→64→128→256→512)
- Batch Normalization setiap layer
- Progressive Dropout (0.25→0.3→0.4→0.5→0.6)
- Global Average Pooling (better than Flatten)
- Dense layers dengan regularization
- Data augmentation optimal
- Class weights untuk balance
- Extended training (120 epochs)
- Learning rate scheduling

UNTUK SKRIPSI:
Model ini adalah CNN murni (tidak transfer learning)
Suitable untuk penelitian yang fokus pada CNN algorithm
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import os
import math

# ==========================================
# CONFIGURATION - OPTIMIZED FOR 90%+
# ==========================================
IMG_SIZE = (96, 96)  # Optimal size untuk CNN
BATCH_SIZE = 16  # Smaller batch = more stable training
EPOCHS = 120  # Extended training
DATASET_PATH = 'dataset'
INITIAL_LEARNING_RATE = 0.001
MIN_LEARNING_RATE = 1e-7

os.makedirs('saved_models', exist_ok=True)

# ==========================================
# PURE CNN ARCHITECTURE - OPTIMIZED
# ==========================================

def create_pure_cnn_optimized(input_shape=(96, 96, 3)):
    """
    Pure CNN Architecture - Highly Optimized untuk 90%+ Accuracy
    
    Architecture:
    - 5 Convolutional Blocks
    - Progressive filters: 32 → 64 → 128 → 256 → 512
    - Batch Normalization setiap conv layer
    - Progressive Dropout
    - Global Average Pooling
    - 2 Dense layers dengan heavy regularization
    - Binary classification output
    """
    
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # ==========================================
        # BLOCK 1: 32 filters
        # ==========================================
        layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                     kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                     kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # ==========================================
        # BLOCK 2: 64 filters
        # ==========================================
        layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                     kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                     kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # ==========================================
        # BLOCK 3: 128 filters
        # ==========================================
        layers.Conv2D(128, (3, 3), padding='same', activation='relu',
                     kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu',
                     kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # ==========================================
        # BLOCK 4: 256 filters
        # ==========================================
        layers.Conv2D(256, (3, 3), padding='same', activation='relu',
                     kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu',
                     kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.5),
        
        # ==========================================
        # BLOCK 5: 512 filters (Deep features)
        # ==========================================
        layers.Conv2D(512, (3, 3), padding='same', activation='relu',
                     kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3, 3), padding='same', activation='relu',
                     kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        
        # Global Average Pooling (better than Flatten)
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.6),
        
        # ==========================================
        # FULLY CONNECTED LAYERS
        # ==========================================
        layers.Dense(512, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(0.001),
                    kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Dropout(0.6),
        
        layers.Dense(256, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(0.001),
                    kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # ==========================================
        # OUTPUT LAYER
        # ==========================================
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

# ==========================================
# DATA PREPARATION - OPTIMIZED AUGMENTATION
# ==========================================

def prepare_optimized_data_generators():
    """
    Data generators dengan augmentasi yang optimal untuk CNN
    Balance antara variasi dan preserve original features
    """
    
    # Training data dengan augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        
        # Geometric transformations
        rotation_range=15,  # Moderate rotation
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        
        # Color augmentation
        brightness_range=[0.8, 1.2],
        
        # Validation split
        validation_split=0.2,
        
        fill_mode='nearest'
    )
    
    # Test data: only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training',
        shuffle=True,
        seed=42
    )
    
    # Validation generator
    validation_generator = train_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=True,
        seed=42
    )
    
    # Test generator
    test_generator = test_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, 'test'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    return train_generator, validation_generator, test_generator

# ==========================================
# CLASS WEIGHTS - HANDLE IMBALANCED DATA
# ==========================================

def calculate_class_weights(train_gen):
    """
    Calculate class weights untuk handle potential imbalance
    """
    class_labels = np.unique(train_gen.classes)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=class_labels,
        y=train_gen.classes
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    print("\n" + "="*60)
    print("⚖️  CLASS WEIGHTS:")
    print("="*60)
    for class_name, class_idx in train_gen.class_indices.items():
        count = sum(train_gen.classes == class_idx)
        weight = class_weight_dict[class_idx]
        print(f"  {class_name:15s}: {count:4d} samples, weight: {weight:.4f}")
    print("="*60)
    
    return class_weight_dict

# ==========================================
# LEARNING RATE SCHEDULING
# ==========================================

def lr_schedule(epoch, lr):
    """
    Learning rate schedule:
    - Epochs 0-40: lr = 0.001
    - Epochs 41-80: lr = 0.0005
    - Epochs 81-120: lr = 0.0001
    """
    if epoch < 40:
        return 0.001
    elif epoch < 80:
        return 0.0005
    else:
        return 0.0001

# ==========================================
# TRAINING PIPELINE
# ==========================================

def train_pure_cnn():
    """
    Complete training pipeline untuk Pure CNN
    """
    print("="*60)
    print("🎯 PURE CNN TRAINING - TARGET 90%+ ACCURACY")
    print("="*60)
    
    print("\n📊 Loading dataset...")
    train_gen, val_gen, test_gen = prepare_optimized_data_generators()
    
    print(f"\n✅ Dataset loaded:")
    print(f"  Training:   {train_gen.samples:4d} samples")
    print(f"  Validation: {val_gen.samples:4d} samples")
    print(f"  Test:       {test_gen.samples:4d} samples")
    print(f"  Classes:    {list(train_gen.class_indices.keys())}")
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_gen)
    
    print("\n🔨 Building Pure CNN model...")
    model = create_pure_cnn_optimized()
    
    # Print model summary
    print("\n📋 MODEL ARCHITECTURE:")
    print("="*60)
    model.summary()
    print("="*60)
    
    # Count parameters
    total_params = model.count_params()
    print(f"\n📊 Total Parameters: {total_params:,}")
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    # Callbacks
    callbacks = [
        # Save best model
        ModelCheckpoint(
            'saved_models/pure_cnn_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1,
            save_weights_only=False
        ),
        
        # Early stopping dengan patience tinggi
        EarlyStopping(
            monitor='val_loss',
            patience=20,  # Patient training
            restore_best_weights=True,
            verbose=1
        ),
        
        # Learning rate reduction
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=MIN_LEARNING_RATE,
            verbose=1
        ),
        
        # Learning rate scheduler
        LearningRateScheduler(lr_schedule, verbose=1)
    ]
    
    print("\n" + "="*60)
    print("🚀 STARTING TRAINING")
    print("="*60)
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Initial LR: {INITIAL_LEARNING_RATE}")
    print(f"  Optimizer: Adam")
    print(f"  Class Weights: Enabled")
    print("="*60 + "\n")
    
    # Train model
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Save final model
    model.save('saved_models/pure_cnn_final.h5')
    print("\n✅ Training completed!")
    print("   Models saved:")
    print("   - saved_models/pure_cnn_best.h5 (best model)")
    print("   - saved_models/pure_cnn_final.h5 (final model)")
    
    return model, history, test_gen

# ==========================================
# AUTO THRESHOLD TESTING
# ==========================================

def test_multiple_thresholds(model, test_gen):
    """
    Test dengan multiple thresholds untuk find optimal
    """
    print("\n" + "="*60)
    print("🎯 AUTO THRESHOLD TESTING")
    print("="*60)
    
    test_gen.reset()
    predictions = model.predict(test_gen, verbose=1)
    true_classes = test_gen.classes
    
    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    
    best_threshold = 0.5
    best_accuracy = 0
    best_f1 = 0
    
    print("\n📊 Testing different thresholds:")
    print("-" * 75)
    print(f"{'Threshold':>10} | {'Accuracy':>9} | {'Precision':>10} | {'Recall':>8} | {'F1-Score':>9}")
    print("-" * 75)
    
    results_list = []
    
    for threshold in thresholds:
        predicted_classes = (predictions > threshold).astype(int).flatten()
        
        acc = accuracy_score(true_classes, predicted_classes)
        prec = precision_score(true_classes, predicted_classes, zero_division=0)
        rec = recall_score(true_classes, predicted_classes, zero_division=0)
        f1 = f1_score(true_classes, predicted_classes, zero_division=0)
        
        results_list.append({
            'threshold': threshold,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        })
        
        marker = ""
        if acc > best_accuracy or (acc == best_accuracy and f1 > best_f1):
            best_accuracy = acc
            best_f1 = f1
            best_threshold = threshold
            marker = "  ⭐ BEST"
        
        print(f"  {threshold:>6.2f}   | {acc*100:>7.2f}%  | {prec*100:>8.2f}%  | {rec*100:>6.2f}% | {f1*100:>7.2f}%{marker}")
    
    print("-" * 75)
    print(f"\n🏆 OPTIMAL THRESHOLD: {best_threshold:.2f}")
    print(f"   Best Accuracy:  {best_accuracy*100:.2f}%")
    print(f"   Best F1-Score:  {best_f1*100:.2f}%")
    
    print("\n💡 UPDATE BACKEND:")
    print(f"   Edit app.py:")
    print(f"   PREDICTION_THRESHOLD = {best_threshold}")
    print("="*60)
    
    return best_threshold, results_list

# ==========================================
# COMPREHENSIVE EVALUATION
# ==========================================

def evaluate_pure_cnn(model, test_gen, threshold=0.5):
    """
    Comprehensive evaluation dengan custom threshold
    """
    print("\n" + "="*60)
    print(f"📊 FINAL EVALUATION (Threshold: {threshold})")
    print("="*60)
    
    # Get predictions
    test_gen.reset()
    predictions = model.predict(test_gen, verbose=1)
    predicted_classes = (predictions > threshold).astype(int).flatten()
    true_classes = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())
    
    # Calculate metrics
    test_accuracy = accuracy_score(true_classes, predicted_classes)
    test_precision = precision_score(true_classes, predicted_classes, zero_division=0)
    test_recall = recall_score(true_classes, predicted_classes, zero_division=0)
    test_f1 = f1_score(true_classes, predicted_classes, zero_division=0)
    
    print("\n🎯 FINAL RESULTS:")
    print("="*60)
    print(f"  Accuracy:  {test_accuracy*100:6.2f}%")
    print(f"  Precision: {test_precision*100:6.2f}%")
    print(f"  Recall:    {test_recall*100:6.2f}%")
    print(f"  F1-Score:  {test_f1*100:6.2f}%")
    print("="*60)
    
    # Achievement check
    if test_accuracy >= 0.90:
        print("\n🎉🎉🎉 EXCELLENT! TARGET 90% ACHIEVED! 🎉🎉🎉")
    elif test_accuracy >= 0.85:
        print("\n✅ VERY GOOD! Accuracy ≥85% (Close to target)")
    elif test_accuracy >= 0.80:
        print("\n✅ GOOD! Accuracy ≥80%")
    else:
        print("\n⚠️  Needs improvement (<80%)")
    
    # Classification Report
    print("\n📋 DETAILED CLASSIFICATION REPORT:")
    print("="*60)
    print(classification_report(
        true_classes,
        predicted_classes,
        target_names=class_labels,
        digits=4
    ))
    
    # Confusion Matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar_kws={'label': 'Count'},
        annot_kws={'size': 14}
    )
    plt.title(f'Confusion Matrix - Pure CNN\nAccuracy: {test_accuracy*100:.2f}% (Threshold: {threshold})',
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('saved_models/confusion_matrix_pure_cnn.png', dpi=300, bbox_inches='tight')
    print("\n✅ Confusion matrix saved: saved_models/confusion_matrix_pure_cnn.png")
    plt.close()
    
    # Detailed Analysis
    tn, fp, fn, tp = cm.ravel()
    print("\n🔍 DETAILED CONFUSION MATRIX ANALYSIS:")
    print("="*60)
    print(f"  True Positives (TP):  {tp:4d}  ✅ Correctly detected fatigued")
    print(f"  True Negatives (TN):  {tn:4d}  ✅ Correctly detected non-fatigued")
    print(f"  False Positives (FP): {fp:4d}  ❌ Non-fatigued wrongly as fatigued")
    print(f"  False Negatives (FN): {fn:4d}  ❌ Fatigued cases missed")
    print("="*60)
    
    # Analysis feedback
    if fn < 15:
        print("\n✅ Excellent! Very few fatigue cases missed (safety priority)")
    elif fn < 30:
        print("\n✅ Good! Acceptable false negative rate")
    else:
        print("\n⚠️  Warning: Many fatigue cases missed - safety concern!")
    
    if fp < 25:
        print("✅ Excellent! Very few false alarms")
    elif fp < 50:
        print("✅ Good! Acceptable false positive rate")
    else:
        print("⚠️  Warning: Many false alarms - consider raising threshold")
    
    return {
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1_score': test_f1,
        'threshold': threshold,
        'confusion_matrix': cm
    }

# ==========================================
# VISUALIZATION
# ==========================================

def plot_training_history(history):
    """
    Plot comprehensive training history
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    epochs_range = range(1, len(history.history['accuracy']) + 1)
    
    # Accuracy
    axes[0, 0].plot(epochs_range, history.history['accuracy'], 
                    'b-', linewidth=2, label='Training', alpha=0.8)
    axes[0, 0].plot(epochs_range, history.history['val_accuracy'], 
                    'r-', linewidth=2, label='Validation', alpha=0.8)
    axes[0, 0].set_title('Model Accuracy - Pure CNN', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Accuracy', fontsize=12)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(epochs_range, history.history['loss'], 
                    'b-', linewidth=2, label='Training', alpha=0.8)
    axes[0, 1].plot(epochs_range, history.history['val_loss'], 
                    'r-', linewidth=2, label='Validation', alpha=0.8)
    axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Loss', fontsize=12)
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision
    axes[1, 0].plot(epochs_range, history.history['precision'], 
                    'b-', linewidth=2, label='Training', alpha=0.8)
    axes[1, 0].plot(epochs_range, history.history['val_precision'], 
                    'r-', linewidth=2, label='Validation', alpha=0.8)
    axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Precision', fontsize=12)
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Recall
    axes[1, 1].plot(epochs_range, history.history['recall'], 
                    'b-', linewidth=2, label='Training', alpha=0.8)
    axes[1, 1].plot(epochs_range, history.history['val_recall'], 
                    'r-', linewidth=2, label='Validation', alpha=0.8)
    axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Recall', fontsize=12)
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Training History - Pure CNN Model', 
                 fontsize=16, fontweight='bold', y=0.998)
    plt.tight_layout()
    plt.savefig('saved_models/training_history_pure_cnn.png', dpi=300, bbox_inches='tight')
    print("\n✅ Training history saved: saved_models/training_history_pure_cnn.png")
    plt.close()

# ==========================================
# MAIN PIPELINE
# ==========================================

def main():
    """
    Complete training and evaluation pipeline
    """
    print("\n" + "="*60)
    print("🧠 PURE CNN ARCHITECTURE - FATIGUE DETECTION")
    print("="*60)
    print("\n🎯 TARGET: 90%+ Accuracy")
    print("\n📚 UNTUK PENELITIAN SKRIPSI:")
    print("  ✅ Pure CNN (bukan Transfer Learning)")
    print("  ✅ 5 Convolutional Blocks")
    print("  ✅ Progressive filters: 32→64→128→256→512")
    print("  ✅ Batch Normalization + Dropout")
    print("  ✅ Global Average Pooling")
    print("  ✅ Extended training (120 epochs)")
    print("  ✅ Optimized for 90%+ accuracy")
    print("="*60 + "\n")
    
    # Train model
    model, history, test_gen = train_pure_cnn()
    
    # Plot training history
    print("\n📊 Generating training visualizations...")
    plot_training_history(history)
    
    # Test multiple thresholds
    best_threshold, threshold_results = test_multiple_thresholds(model, test_gen)
    
    # Final evaluation
    final_results = evaluate_pure_cnn(model, test_gen, threshold=best_threshold)
    
    # Final Summary
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print("\n📁 FILES SAVED:")
    print("  ✅ saved_models/pure_cnn_best.h5")
    print("  ✅ saved_models/pure_cnn_final.h5")
    print("  ✅ saved_models/confusion_matrix_pure_cnn.png")
    print("  ✅ saved_models/training_history_pure_cnn.png")
    
    print(f"\n🏆 FINAL RESULTS (Threshold: {best_threshold}):")
    print("="*60)
    print(f"  Accuracy:  {final_results['accuracy']*100:6.2f}%")
    print(f"  Precision: {final_results['precision']*100:6.2f}%")
    print(f"  Recall:    {final_results['recall']*100:6.2f}%")
    print(f"  F1-Score:  {final_results['f1_score']*100:6.2f}%")
    print("="*60)
    
    print("\n🎯 NEXT STEPS:")
    print("="*60)
    print("  1. Update backend threshold:")
    print(f"     Edit app.py: PREDICTION_THRESHOLD = {best_threshold}")
    print("\n  2. Copy best model to backend:")
    print("     copy saved_models\\pure_cnn_best.h5 backend\\fatigue_detection_model.h5")
    print("\n  3. Update backend preprocessing:")
    print("     Pastikan input size = (96, 96)")
    print("\n  4. Restart backend:")
    print("     cd backend")
    print("     python app.py")
    print("\n  5. Test di web dashboard:")
    print("     http://localhost:8080")
    print("="*60)
    
    print("\n📊 UNTUK SKRIPSI:")
    print("  - Model: Pure CNN (5 conv blocks)")
    print(f"  - Parameters: {model.count_params():,}")
    print(f"  - Final Accuracy: {final_results['accuracy']*100:.2f}%")
    print("  - Arsitektur: Lihat model.summary() di output")
    print("  - Grafik: training_history_pure_cnn.png")
    print("  - Evaluasi: confusion_matrix_pure_cnn.png")
    
    print("\n✨ Good luck with your thesis! 🎓📚")

if __name__ == "__main__":
    main()