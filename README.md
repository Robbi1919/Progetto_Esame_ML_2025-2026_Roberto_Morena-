Startup Success Prediction - Machine Learning Project
Progetto di Machine Learning per la predizione del successo delle startup studentesche nei programmi di incubazione universitaria.
Autore: Roberto Morena  
Matricola: 0212712788  
Corso: Machine Learning - A.A. 2025-2026
---
## Descrizione

Questo progetto sviluppa un sistema di machine learning per predire il successo di startup studentesche utilizzando dati provenienti da programmi di incubazione universitari. L'obiettivo è supportare decisioni strategiche di allocazione risorse (mentorship, funding, supporto incubazione) basate su analisi quantitative.

Il modello Random Forest ottimale raggiunge un'accuratezza del 92.38% sul test set, superando l'obiettivo minimo dell'85%.

## Dataset
Il dataset "Student Startup Success" contiene 2,100 progetti imprenditoriali studenteschi da 40 università nel periodo 2019-2023.

**Caratteristiche principali:**
- 2,100 record totali
- 16 colonne (11 numeriche, 3 categoriche, 1 target binario)
- 0 valori mancanti (completezza 100%)
- 0 outliers rilevati
- Distribuzione target: 58% fallimenti, 42% successi

**Features principali:**
- team_size: dimensione del team
- avg_team_experience: esperienza media del team (anni)
- innovation_score: punteggio innovazione (0-1)
- funding_amount_usd: ammontare finanziamento in dollari
- mentorship_support: presenza supporto mentorship (0/1)
- incubation_support: presenza supporto incubatore (0/1)
- market_readiness_level: livello maturità mercato (0-1)
- competition_awards: premi vinti in competizioni
- business_model_score: punteggio modello di business (0-1)
- technology_maturity: maturità tecnologica (0-1)
- institution_type: tipologia istituzione (Public/Private/Technical/Non-technical)
- project_domain: settore startup (AgriTech/FinTech/GreenTech/HealthTech/EdTech)
- success_label: successo/fallimento (0/1)

---

## Pipeline
### 1. Analisi Esplorativa (EDA)
**Missing Values:** Nessun valore mancante rilevato.
**Outliers:** Applicato metodo IQR su 11 features numeriche. Nessun outlier identificato.
**Correlazioni:** Analisi correlazione di Pearson. Top 3 features correlate con il successo:
- funding_amount_usd (r=0.395)
- mentorship_support (r=0.373)
- incubation_support (r=0.367)
**Features Categoriche:** Test Chi-squared per indipendenza. Nessuna associazione statisticamente significativa con il target (p>0.05), ma mantenute per catturare interazioni non-lineari.

### 2. Preprocessing
**One-Hot Encoding:** Le 3 features categoriche sono state trasformate in 46 colonne binarie (drop_first=True).
**Train/Test Split:** Split stratificato 80/20 con random_state=42
- Training set: 1,680 campioni (58.0% fallimenti, 42.0% successi)
- Test set: 420 campioni (58.1% fallimenti, 41.9% successi)
**Class Imbalance:** Gestito tramite class_weight='balanced' (pesi: 0.86 fallimenti, 1.19 successi).
**Dataset finale:** 57 features (11 numeriche + 46 encoded)

### 3. Modelli Implementati
**Random Forest Classifier**
- 100 alberi, profondità massima 20, features casuali per split
- Training time: 1.08 secondi
**Decision Tree Classifier**
- Profondità massima 10, criteri anti-overfitting
- Training time: 0.014 secondi
**Gaussian Naive Bayes**
- Modello probabilistico basato su Teorema di Bayes
- Training time: 0.016 secondi

### 4. Validazione
**Cross-Validation:** 5-Fold Stratified Cross-Validation per valutare stabilità e generalizzazione.
**Metriche:** Accuracy, Precision, Recall, F1-Score, ROC-AUC

## Risultati

### Performance Comparativa
| Modello | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------|----------|-----------|--------|----------|---------|
| Random Forest | 92.38% | 0.9138 | 0.9034 | 0.9086 | 0.9785 |
| Naive Bayes | 80.48% | 0.7304 | 0.8466 | 0.7842 | 0.9051 |
| Decision Tree | 66.43% | 0.5967 | 0.6136 | 0.6050 | 0.7228 |

### Cross-Validation (5-Fold)
| Modello | CV Accuracy (media ± std) | Gap Test-CV |
|---------|---------------------------|-------------|
| Random Forest | 92.98% ± 1.37% | 0.60% |
| Naive Bayes | 83.57% ± 2.08% | 3.09% |
| Decision Tree | 70.00% ± 2.95% | 3.57% |

### Confusion Matrix - Random Forest
- True Negatives: 229 (fallimenti identificati correttamente)
- True Positives: 159 (successi identificati correttamente)
- False Positives: 15 (predetti successo ma falliti)
- False Negatives: 17 (predetti fallimento ma riusciti)
- Errori totali: 32/420 (7.6%)

### Feature Importance (Top 5)
Le features più importanti identificate dal modello Random Forest:

1. funding_amount_usd: 21.33%
2. innovation_score: 15.63%
3. mentorship_support: 15.32%
4. incubation_support: 14.78%
5. business_model_score: 10.19%

Queste 5 features contribuiscono al 77.24% delle decisioni del modello.

**Insight pratico:** Il finanziamento è il predittore dominante del successo, seguito dall'innovazione e dal supporto di mentorship/incubazione.

## Struttura Repository

