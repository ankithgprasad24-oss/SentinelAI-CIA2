# SentinelAI — Practical Demonstration Script

## 1. Introduction (30 seconds)
“Good morning. My project is SentinelAI, an AI-powered fraud detection prototype. It analyses transaction patterns and flags unusual activity for human review.”

## 2. Explain the problem (30 seconds)
“Traditional fixed rules may miss combinations of unusual behaviour. SentinelAI demonstrates anomaly detection by learning a reference pattern from synthetic transaction data.”

## 3. Demonstrate the application (2 minutes)
1. Open the Overview page and explain the dashboard metrics.
2. Open AI Scanner.
3. Enter a typical transaction, for example amount ₹1,500, hour 14, Grocery, distance 2 km, one transaction in the last hour, and no international activity.
4. Run the analysis and explain the anomaly indicator.
5. Enter an unusual example, for example amount ₹75,000, hour 2, Electronics, distance 180 km, six transactions in the last hour, and international activity.
6. Run the analysis and compare the result.
7. Open CSV Analysis, download the sample CSV, upload it, and show the result table and export option.
8. Open Transaction History and show filtering/export.

## 4. Explain the AI (1 minute)
“SentinelAI uses Isolation Forest, an unsupervised machine-learning algorithm. It creates random isolation trees. Unusual observations tend to be isolated in fewer splits. We transform numeric features with StandardScaler and encode merchant categories using OneHotEncoder. The model's anomaly score is compared with a threshold calibrated on synthetic reference data.”

## 5. Limitations (30 seconds)
“The sample data is synthetic, so this project demonstrates the method rather than claiming real-world fraud-detection accuracy. An anomaly is not proof of fraud. A real deployment would require approved, representative, privacy-protected transaction data, validation, monitoring, and human review.”

## 6. Closing
“SentinelAI demonstrates how machine learning can support financial security by prioritising unusual transactions for investigation.”
