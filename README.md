# Dark Patterns und Social Media Analyse 🕵️‍♂️📱

**Wie Design Nutzerverhalten, Suchtentwicklung und Kaufentscheidungen beeinflusst.**

> *Projektarbeit an der FH Oberösterreich | Studiengang: Mobile Computing*

## 📖 Über das Projekt
Dieses Projekt untersucht die Schnittstelle zwischen psychologischen Mechanismen und technischem Design. Wir analysieren zwei zentrale Aspekte des digitalen Raums:
1.  **Social Media:** Wie nutzen Menschen Plattformen und was lässt sich daraus bezüglich Suchtpotenzial ableiten?
2.  **Dark Patterns:** In welchem Ausmaß und mit welchen Mitteln werden Nutzer im E-Commerce manipuliert?

Ziel ist es, manipulative Designmuster nicht nur theoretisch zu beschreiben, sondern empirisch sichtbar zu machen.

## 📊 Zentrale Ergebnisse

### 1. Dark Patterns im E-Commerce
Basierend auf der Analyse von ~11.000 Shopping-Webseiten:
* **Verbreitung:** Über **12,71 %** aller untersuchten Webseiten nutzen mindestens ein Dark Pattern.
* **Top-Kategorien:** Die häufigsten manipulativen Muster sind *Scarcity* (Verknappung), *Urgency* (Dringlichkeit) und *Social Proof*.
* **Popularität:** Bekanntere Webseiten (laut Alexa-Ranking) setzen signifikant häufiger manipulative Muster ein.

### 2. Social Media Nutzerverhalten
Mittels K-Means Clustering und PCA konnten wir drei spezifische Nutzertypen identifizieren:
* 🔴 **Impulsive Dauernutzer:** Hoher Konsum bei geringer Selbstregulation.
* 🔵 **Unzufriedene Gelegenheitsnutzer:** Mittlere Nutzung, aber geringe Zufriedenheit.
* 🟢 **Kontrollierte Vielnutzer:** Hohe Nutzung bei gleichzeitig hoher Selbstkontrolle.

**Überraschende Erkenntnis:** Die Nutzungsdauer ist über alle Altersgruppen (18 bis 65+) hinweg relativ homogen. Ältere Nutzer verbringen fast ebenso viel Zeit auf den Plattformen wie die Gruppe der 18- bis 25-Jährigen.

## 💾 Datensätze
Das Projekt verwendet zwei externe Datensätze:
* **Dark Patterns Dataset:** Stammt von Mathur et al. (Princeton University) und enthält Crawling-Daten von Shopping-Seiten.
* **Time Wasters on Social Media:** Ein Datensatz von Kaggle (Zeesolver), der Nutzungsdauer, Plattformen und Suchtindikatoren verknüpft.

## 🛠 Technologien
* **Python** (Datenanalyse & Visualisierung)
* **Pandas & NumPy** (Datenverarbeitung)
* **Seaborn & Matplotlib** (Visualisierung der Cluster und Verteilungen)
* **Scikit-Learn** (K-Means Clustering, PCA)

## 👥 Autoren
* **Niklas Gottlieb-Zimmermann**
* **Raphael Gruber**

---
*Disclaimer: Dieses Repository dient akademischen Zwecken zur Aufklärung über digitale Manipulation.*
