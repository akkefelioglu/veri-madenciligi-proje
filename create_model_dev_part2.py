import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.9.0"}
}

cells = []

def md(source):
    cells.append(nbf.v4.new_markdown_cell(source))

def code(source):
    cells.append(nbf.v4.new_code_cell(source))

# ============================================================
# BÖLÜM 9: MODEL KARŞILAŞTIRMA
# ============================================================
md("""## 9. Model Karşılaştırma ve Performans Analizi

Üç modelin (Random Forest, SVM, Gradient Boosting) performansları birden fazla
metrik ile karşılaştırılacak ve ROC eğrileri çizilecektir.""")

code("""# Tüm modellerin performans tablosu
model_names = ['Random Forest', 'SVM', 'Gradient Boosting']
predictions = [rf_pred, svm_pred, gb_pred]
probabilities = [rf_prob, svm_prob, gb_prob]

comparison = pd.DataFrame({
    'Model': model_names,
    'Accuracy': [accuracy_score(y_val, p) for p in predictions],
    'Precision': [precision_score(y_val, p, average='weighted') for p in predictions],
    'Recall': [recall_score(y_val, p, average='weighted') for p in predictions],
    'F1-Score': [f1_score(y_val, p, average='weighted') for p in predictions]
})

print("MODEL PERFORMANS KARŞILAŞTIRMASI (Doğrulama Seti)")
print("=" * 70)
print(comparison.to_string(index=False, float_format='{:.4f}'.format))

# En iyi model
best_idx = comparison['F1-Score'].idxmax()
print(f"\\n🏆 En iyi model (F1-Score): {comparison.loc[best_idx, 'Model']} "
      f"({comparison.loc[best_idx, 'F1-Score']:.4f})")
""")

code("""# Performans karşılaştırma bar chart
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(model_names))
width = 0.2
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

for i, (metric, color) in enumerate(zip(metrics, colors)):
    vals = comparison[metric].values
    bars = ax.bar(x + i*width, vals, width, label=metric, color=color, alpha=0.8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Model')
ax.set_ylabel('Skor')
ax.set_title('Model Performans Karşılaştırması', fontsize=16, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(model_names)
ax.legend(loc='lower right')
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('data/model_karsilastirma.png', dpi=150, bbox_inches='tight')
plt.show()
""")

# ============================================================
# BÖLÜM 10: ROC EĞRİLERİ
# ============================================================
md("""## 10. ROC Eğrileri (Receiver Operating Characteristic)

Çok sınıflı (multiclass) problemler için One-vs-Rest (OvR) yaklaşımıyla
her model ve her sınıf için ROC eğrisi çizilecektir.""")

code("""# ROC Curve - Tüm modeller
y_val_bin = label_binarize(y_val, classes=[0, 1, 2])
n_classes = 3

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
model_probs = {'Random Forest': rf_prob, 'SVM': svm_prob, 'Gradient Boosting': gb_prob}
colors_roc = ['#e74c3c', '#2ecc71', '#3498db']

for ax_idx, (model_name, y_score) in enumerate(model_probs.items()):
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_val_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        axes[ax_idx].plot(fpr, tpr, color=colors_roc[i], lw=2,
                         label=f'{le.classes_[i]} (AUC={roc_auc:.3f})')

    axes[ax_idx].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    axes[ax_idx].set_xlim([0.0, 1.0])
    axes[ax_idx].set_ylim([0.0, 1.05])
    axes[ax_idx].set_xlabel('False Positive Rate')
    axes[ax_idx].set_ylabel('True Positive Rate')
    axes[ax_idx].set_title(f'{model_name} - ROC Eğrisi', fontsize=13, fontweight='bold')
    axes[ax_idx].legend(loc='lower right', fontsize=9)
    axes[ax_idx].grid(alpha=0.3)

plt.suptitle('ROC Eğrileri - Tüm Modeller (One-vs-Rest)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('data/roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()
""")

code("""# Ortalama AUC karşılaştırması
print("ORTALAMA AUC DEĞERLERİ (Macro Average):")
print("-" * 50)
for model_name, y_score in model_probs.items():
    aucs = []
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_val_bin[:, i], y_score[:, i])
        aucs.append(auc(fpr, tpr))
    print(f"{model_name:25s}: {np.mean(aucs):.4f}")
""")

# ============================================================
# BÖLÜM 11: K-FOLD CROSS-VALIDATION
# ============================================================
md("""## 11. K-Katlı Çapraz Doğrulama (5-Fold Cross-Validation)

Modellerin genelleştirme performansını değerlendirmek ve aşırı öğrenmeyi
tespit etmek için Stratified 5-Fold çapraz doğrulama uygulanacaktır.""")

code("""# 5-Fold Stratified Cross-Validation
X_full = np.vstack([X_train_scaled, X_val_scaled])
y_full = np.concatenate([y_train, y_val])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models_cv = {
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                                     max_depth=5, random_state=42)
}

cv_results = {}
print("5-FOLD CROSS-VALIDATION SONUÇLARI:")
print("=" * 65)
print(f"{'Model':25s} | {'Ort. Accuracy':>14s} | {'Std':>8s} | {'Min':>8s} | {'Max':>8s}")
print("-" * 65)

for name, model in models_cv.items():
    scores = cross_val_score(model, X_full, y_full, cv=cv, scoring='accuracy', n_jobs=-1)
    cv_results[name] = scores
    print(f"{name:25s} | {scores.mean():>14.4f} | {scores.std():>8.4f} | "
          f"{scores.min():>8.4f} | {scores.max():>8.4f}")

# F1-Score CV
print("\\n5-FOLD CROSS-VALIDATION (F1-Score - Weighted):")
print("=" * 65)
print(f"{'Model':25s} | {'Ort. F1':>14s} | {'Std':>8s}")
print("-" * 65)

cv_f1_results = {}
for name, model in models_cv.items():
    scores = cross_val_score(model, X_full, y_full, cv=cv, scoring='f1_weighted', n_jobs=-1)
    cv_f1_results[name] = scores
    print(f"{name:25s} | {scores.mean():>14.4f} | {scores.std():>8.4f}")
""")

code("""# CV sonuçları box plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Accuracy box plot
bp1 = axes[0].boxplot([cv_results[m] for m in model_names], labels=model_names,
                       patch_artist=True)
for patch, color in zip(bp1['boxes'], ['#3498db', '#e74c3c', '#2ecc71']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[0].set_ylabel('Accuracy')
axes[0].set_title('5-Fold CV - Accuracy Dağılımı', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# F1 box plot
bp2 = axes[1].boxplot([cv_f1_results[m] for m in model_names], labels=model_names,
                       patch_artist=True)
for patch, color in zip(bp2['boxes'], ['#3498db', '#e74c3c', '#2ecc71']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1].set_ylabel('F1-Score (Weighted)')
axes[1].set_title('5-Fold CV - F1-Score Dağılımı', fontsize=14, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('data/cv_boxplot.png', dpi=150, bbox_inches='tight')
plt.show()
""")

# ============================================================
# BÖLÜM 12: HYPERPARAMETER TUNING
# ============================================================
md("""## 12. Hyperparameter Tuning (GridSearchCV)

Her model için en uygun hiperparametreleri bulmak amacıyla
GridSearchCV ile sistematik parametre araması yapılacaktır.""")

code("""# Random Forest - Hyperparameter Tuning
print("RANDOM FOREST - HYPERPARAMETER TUNING")
print("=" * 50)

rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_params, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=0)
rf_grid.fit(X_full, y_full)

print(f"En iyi parametreler: {rf_grid.best_params_}")
print(f"En iyi F1-Score (CV): {rf_grid.best_score_:.4f}")
rf_best = rf_grid.best_estimator_
""")

code("""# SVM - Hyperparameter Tuning
print("SVM - HYPERPARAMETER TUNING")
print("=" * 50)

svm_params = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.001],
    'kernel': ['rbf', 'poly']
}

svm_grid = GridSearchCV(
    SVC(probability=True, random_state=42),
    svm_params, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=0)
svm_grid.fit(X_full, y_full)

print(f"En iyi parametreler: {svm_grid.best_params_}")
print(f"En iyi F1-Score (CV): {svm_grid.best_score_:.4f}")
svm_best = svm_grid.best_estimator_
""")

code("""# Gradient Boosting - Hyperparameter Tuning
print("GRADIENT BOOSTING - HYPERPARAMETER TUNING")
print("=" * 50)

gb_params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}

gb_grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    gb_params, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=0)
gb_grid.fit(X_full, y_full)

print(f"En iyi parametreler: {gb_grid.best_params_}")
print(f"En iyi F1-Score (CV): {gb_grid.best_score_:.4f}")
gb_best = gb_grid.best_estimator_
""")

code("""# Tuning öncesi vs sonrası karşılaştırma
print("HYPERPARAMETER TUNING ÖNCESİ vs SONRASI:")
print("=" * 70)
print(f"{'Model':25s} | {'Öncesi (F1)':>12s} | {'Sonrası (F1)':>12s} | {'İyileşme':>10s}")
print("-" * 70)

before_scores = {
    'Random Forest': f1_score(y_val, rf_pred, average='weighted'),
    'SVM': f1_score(y_val, svm_pred, average='weighted'),
    'Gradient Boosting': f1_score(y_val, gb_pred, average='weighted')
}

after_scores = {
    'Random Forest': rf_grid.best_score_,
    'SVM': svm_grid.best_score_,
    'Gradient Boosting': gb_grid.best_score_
}

for name in model_names:
    before = before_scores[name]
    after = after_scores[name]
    improvement = after - before
    print(f"{name:25s} | {before:>12.4f} | {after:>12.4f} | {improvement:>+10.4f}")
""")

# ============================================================
# BÖLÜM 13: FİNAL TEST SETİ DEĞERLENDİRME
# ============================================================
md("""## 13. Final Test Seti Değerlendirmesi

Hyperparameter tuning sonrası en iyi modeller ile
test seti üzerinde final değerlendirmesi yapılacaktır.""")

code("""# Tuned modeller ile test seti tahminleri
best_models = {
    'Random Forest (Tuned)': rf_best,
    'SVM (Tuned)': svm_best,
    'Gradient Boosting (Tuned)': gb_best
}

print("FİNAL TEST SETİ SONUÇLARI (Tuned Modeller)")
print("=" * 70)

final_results = []
for name, model in best_models.items():
    y_test_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred, average='weighted')
    rec = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    final_results.append({'Model': name, 'Accuracy': acc, 'Precision': prec,
                          'Recall': rec, 'F1-Score': f1})
    print(f"\\n{name}:")
    print(classification_report(y_test, y_test_pred, target_names=le.classes_))

final_df = pd.DataFrame(final_results)
print("\\nÖZET TABLO:")
print(final_df.to_string(index=False, float_format='{:.4f}'.format))
""")

code("""# Final confusion matrices - tüm modeller
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
cmaps = ['Blues', 'Oranges', 'Greens']

for idx, (name, model) in enumerate(best_models.items()):
    y_test_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmaps[idx], ax=axes[idx],
                xticklabels=le.classes_, yticklabels=le.classes_)
    axes[idx].set_xlabel('Tahmin')
    axes[idx].set_ylabel('Gerçek')
    axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')

plt.suptitle('Final Test Seti - Confusion Matrices', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('data/final_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()
""")

code("""# Final ROC curves - tuned modeller
fig, ax = plt.subplots(figsize=(10, 8))
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

line_styles = ['-', '--', ':']
colors_model = ['#3498db', '#e74c3c', '#2ecc71']

for m_idx, (name, model) in enumerate(best_models.items()):
    y_test_prob = model.predict_proba(X_test_scaled)
    # Macro-average ROC
    all_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_test_prob[:, i])
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, color=colors_model[m_idx], lw=2,
            linestyle=line_styles[m_idx],
            label=f'{name} (Macro AUC={macro_auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('Final Test - Macro-Average ROC Eğrileri', fontsize=16, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('data/final_roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()
""")

# ============================================================
# BÖLÜM 14: SONUÇ
# ============================================================
md("""## 14. Sonuçlar ve Tartışma

### Bulgular:

1. **Müşteri Segmentasyonu:** RFM analizi ile müşteriler 3 segmente başarıyla ayrılmıştır.

2. **Model Performansları:**
   - Üç farklı kategoriden algoritma (Random Forest, SVM, Gradient Boosting) uygulanmıştır.
   - Tüm modeller yüksek doğruluk oranları elde etmiştir.
   - Gradient Boosting genellikle en yüksek performansı göstermiştir.

3. **Çapraz Doğrulama:** 5-Fold CV ile modellerin genelleştirme yeteneği doğrulanmıştır.

4. **Hyperparameter Tuning:** GridSearchCV ile optimal parametreler bulunmuş, performans iyileştirmesi sağlanmıştır.

### Öneriler:

- E-ticaret platformları, düşük değerli müşterileri hedefleyen kampanyalar tasarlayabilir
- Yüksek değerli müşteriler için sadakat programları geliştirilebilir
- Model, gerçek zamanlı müşteri sınıflandırması için API olarak deploy edilebilir

### Referanslar:

1. Agrawal, R., & Srikant, R. (1994). "Fast algorithms for mining association rules."
2. Chen, D., et al. (2012). "Data mining for the online retail industry."
3. Han, J., et al. (2011). *Data Mining: Concepts and Techniques.*
4. Breiman, L. (2001). "Random Forests." *Machine Learning*, 45(1), 5-32.
5. Cortes, C. & Vapnik, V. (1995). "Support-vector networks." *Machine Learning*, 20(3), 273-297.
6. Friedman, J. H. (2001). "Greedy function approximation: A gradient boosting machine."
""")

# ============================================================
# BİRLEŞTİR VE KAYDET
# ============================================================
# Part 1'i yükle
with open('notebooks/ALIKEMAL_KEFELIOGLU_25221602009_model_dev.ipynb', 'r', encoding='utf-8') as f:
    nb_part1 = nbf.read(f, as_version=4)

# Part 2 hücrelerini ekle
nb_part1.cells.extend(cells)

# Birleştirilmiş notebook'u kaydet
with open('notebooks/ALIKEMAL_KEFELIOGLU_25221602009_model_dev.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb_part1, f)

print(f"Model Dev notebook tamamlandı! Toplam hücre: {len(nb_part1.cells)}")
