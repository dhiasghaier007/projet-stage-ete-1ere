#!/usr/bin/env python3
"""
Generate 5 sample test documents with REAL multilingual content.
Each file contains a full document with multiple paragraphs, not tiny snippets.
"""

import json
from pathlib import Path
from datetime import datetime

# Create test output directory
TEST_DIR = Path("./test_multilingual_samples")
TEST_DIR.mkdir(exist_ok=True)

# Define 5 carefully crafted test documents with real content
TEST_DOCUMENTS = [
    {
        "filename": "doc_001_hr_policy_english.txt",
        "true_department": "HR",
        "true_doc_type": "Policy",
        "true_language": "EN",
        "true_sensitivity": "Confidential",
        "content": """REMOTE WORK POLICY — Effective January 2026

1. OVERVIEW
The Human Resources department has established a flexible remote work policy to support employee productivity, work-life balance, and access to talent regardless of geographic location. This policy applies to all full-time and part-time employees in the HR department.

2. ELIGIBILITY
Employees must have been employed for a minimum of 90 days and have a satisfactory performance record. Certain roles requiring physical presence in the office (receptionist, facilities coordinator) are not eligible for remote work.

3. WORK SCHEDULE
Approved remote workers may work from home up to 3 days per week, with a minimum 2 days per week in the office for team collaboration and meetings. All remote employees must maintain core business hours from 9:00 AM to 3:00 PM local time.

4. EQUIPMENT AND SECURITY
The company will provide necessary equipment for remote work, including laptop, monitor, and keyboard. All remote workers must use VPN for company network access and maintain current antivirus software.

5. APPROVAL AND REVOCATION
Remote work requests must be submitted through the HR Portal and approved by both the direct manager and HR Director. The company reserves the right to revoke remote work privileges if performance standards are not maintained or if business needs change.

6. EFFECTIVE DATE
This policy is effective immediately for all new requests as of January 15, 2026."""
    },
    {
        "filename": "doc_002_finance_invoice_french.txt",
        "true_department": "Finance",
        "true_doc_type": "Invoice",
        "true_language": "FR",
        "true_sensitivity": "Internal",
        "content": """FACTURE PROFESSIONNELLE

Numéro de Facture: INV-2026-001547
Date de Facture: 2026-01-20
Date d'Échéance: 2026-02-20

FACTURÉ À:
Acme Corporation
Département Finance
123 Avenue Principale
Montréal, QC H1A 1A1

DESCRIPTION DES SERVICES:
Nous vous remercions de votre engagement. Voici un résumé des services professionnels fournis pour le trimestre Q1 2026:

- Consultation stratégique financière (16 heures): Services d'expertise couvrant l'analyse budgétaire, la planification financière et l'optimisation des dépenses opérationnelles. Les consultants ont examiné les dépenses départementales et fourni des recommandations d'économie.

- Audit de conformité (24 heures): Vérification complète des processus financiers et de la conformité réglementaire pour la période janvier-mars 2026.

- Rapports trimestriels (8 heures): Préparation et présentation des rapports financiers détaillés pour la direction générale.

MONTANTS:
Consultation stratégique (16 h × 150$/h): 2,400.00 $
Audit de conformité (24 h × 150$/h): 3,600.00 $
Rapports trimestriels (8 h × 150$/h): 1,200.00 $

Sous-total: 7,200.00 $
TPS (5%): 360.00 $
TVQ (20%): 1,440.00 $

MONTANT TOTAL DÛ: 9,000.00 $

CONDITIONS DE PAIEMENT:
Veuillez remiser le paiement au compte de banque fourni ci-joint ou par chèque adressé au Département Finance. Aucun escompte pour paiement anticipé n'est disponible."""
    },
    {
        "filename": "doc_003_legal_contract_arabic.txt",
        "true_department": "Legal",
        "true_doc_type": "Contract",
        "true_language": "AR",
        "true_sensitivity": "Restricted",
        "content": """اتفاقية خدمة شاملة

تم توقيع هذه الاتفاقية في: 2026-01-15
تاريخ بدء النفاذ: 2026-02-01

الأطراف:
- الشركة الأولى: شركتنا ("الشركة")
- الطرف الثاني: شركة تقديم الخدمات ("المزود")

1. نطاق الخدمات
يوافق المزود على تقديم خدمات استشارية متخصصة في مجال تكنولوجيا المعلومات. تشمل الخدمات ما يلي:
- إدارة البنية التحتية للأنظمة
- دعم تقني على مدار الساعة
- تحديثات الأمان والصيانة الدورية
- التقارير الشهلية عن الأداء

2. المدة والتجديد
تستمر هذه الاتفاقية لمدة سنة واحدة من تاريخ البدء. يمكن تجديد الاتفاقية بموافقة خطية من الطرفين.

3. الالتزامات المالية
ستدفع الشركة مبلغاً شهرياً قدره 5000 دولار أمريكي للخدمات المذكورة أعلاه. يتم الدفع في غضون 30 يوم من استلام الفاتورة.

4. السرية والحماية
يوافق المزود على الحفاظ على سرية جميع المعلومات التجارية والتقنية للشركة. لا يمكن الكشف عن هذه المعلومات لأي طرف ثالث دون موافقة خطية مسبقة.

5. شروط الإنهاء
يحق لأي طرف إنهاء هذه الاتفاقية بموافقة 60 يوم مسبقة. في حالة الانتهاء قبل انتهاء المدة، سيتم استرجاع المبالغ المدفوعة مقابل الخدمات المتبقية.

6. المسؤولية والتعويضات
لن تكون أي من الأطراف مسؤولة عن الأضرار غير المتوقعة أو المباشرة الناشئة عن هذه الاتفاقية."""
    },
    {
        "filename": "doc_004_it_report_english.txt",
        "true_department": "IT",
        "true_doc_type": "Report",
        "true_language": "EN",
        "true_sensitivity": "Internal",
        "content": """SYSTEM UPTIME AND PERFORMANCE REPORT
Week 3 — January 15-21, 2026

EXECUTIVE SUMMARY
Overall system availability for the week was 99.87%, with 2 incidents requiring intervention. Both incidents were resolved within agreed service level agreements. Network performance remained stable, and all critical infrastructure maintained optimal performance.

1. UPTIME METRICS
- Email Systems: 99.98% uptime (1 minute of unplanned downtime)
- File Servers: 99.95% uptime (4 minutes of maintenance downtime)
- Web Applications: 99.87% uptime (18 minutes total — see incident report below)
- Database Infrastructure: 100% uptime (all maintenance windows completed Friday night)
- Network Core: 99.99% uptime

2. INCIDENT REPORTS
Incident #437: Database Query Performance Degradation (Jan 17, 2:45 PM - 3:00 PM)
- Impact: Finance department reporting queries running 10x slower than baseline
- Root Cause: Index fragmentation on transaction_history table
- Resolution: Executed maintenance job to rebuild indexes
- Duration: 15 minutes
- SLA Status: Resolved within 4-hour response time ✓

Incident #438: Load Balancer Configuration Error (Jan 20, 11:30 AM - 11:48 AM)
- Impact: 15% of users experienced intermittent connection failures to web portal
- Root Cause: Incorrect failover configuration deployed during Thursday update
- Resolution: Rollback to previous configuration, re-tested failover logic
- Duration: 18 minutes
- SLA Status: Resolved within 4-hour response time ✓

3. CAPACITY AND UTILIZATION
- Storage Utilization: 68% (within normal parameters)
- Database CPU Average: 42% (healthy baseline)
- Memory Utilization: 55% across production servers
- Network Bandwidth Peak: 72% during business hours

4. SECURITY UPDATES
All critical security patches were applied during scheduled maintenance windows. No vulnerabilities remain unpatched beyond their grace period.

5. RECOMMENDATIONS FOR NEXT WEEK
- Monitor index fragmentation on newly optimized tables
- Conduct load balancer failover drill during maintenance window
- Plan capacity upgrade for storage to bring utilization below 70%"""
    },
    {
        "filename": "doc_005_general_email_french_arabic.txt",
        "true_department": "General",
        "true_doc_type": "Email",
        "true_language": "FR",
        "true_sensitivity": "Public",
        "content": """De: direction@entreprise.com
À: tous.employes@entreprise.com
Date: 2026-01-22
Objet: Annonce Importante — Déjeuner d'Équipe et Fermeture du Bureau

Bonjour à tous,

Nous vous écrivons pour annoncer deux événements importants pour notre organisation:

1. DÉJEUNER D'ÉQUIPE TRIMESTRIAL
Nous sommes heureux d'annoncer notre déjeuner d'équipe du trimestre prochain, qui aura lieu ce vendredi 24 janvier 2026 à 12h00 dans la grande salle de conférence du rez-de-chaussée.

Le menu inclura plusieurs options pour accommoder toutes les préférences alimentaires:
- Plats végétariens et véganiens
- Protéines sans gluten
- Options de fruits de mer
- Desserts variés

Veuillez confirmer votre présence avant mercredi 22 janvier via le système de réservation en ligne. Le nombre de participants nous aide à préparer suffisamment de nourriture pour tous.

2. FERMETURE DU BUREAU
S'il vous plaît, veuillez noter que le bureau fermera plus tôt le mardi 28 janvier 2026 à 15h00 pour permettre la maintenance annuelle de nos systèmes informatiques, l'installation de nouveaux équipements, et le nettoyage en profondeur des installations.

Tous les employés doivent planifier de terminer leur travail essentiel avant cette heure. Les systèmes informatiques seront hors ligne entre 15h00 et 22h00. Le bureau réouvrira à ses heures normales le mercredi 29 janvier.

Pour toute question ou préoccupation, n'hésitez pas à contacter le département des Ressources Humaines.

Cordialement,
La Direction"""
    }
]

def generate_test_files():
    """Write the 5 test documents to files and create metadata."""
    metadata = []
    
    for doc in TEST_DOCUMENTS:
        filepath = TEST_DIR / doc["filename"]
        filepath.write_text(doc["content"], encoding="utf-8")
        print(f"✓ Created: {doc['filename']} ({doc['true_language']})")
        
        # Build metadata entry (exclude content)
        meta_entry = {k: v for k, v in doc.items() if k != "content"}
        meta_entry["filename"] = doc["filename"]
        metadata.append(meta_entry)
    
    # Write metadata file
    metadata_file = TEST_DIR / "metadata.jsonl"
    with open(metadata_file, "w", encoding="utf-8") as f:
        for entry in metadata:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"\n✓ Metadata written to: {metadata_file.resolve()}")
    print(f"\nTest corpus ready at: {TEST_DIR.resolve()}")
    print("\nSummary:")
    print(f"  - {sum(1 for d in TEST_DOCUMENTS if d['true_language'] == 'EN')} English documents")
    print(f"  - {sum(1 for d in TEST_DOCUMENTS if d['true_language'] == 'FR')} French documents")
    print(f"  - {sum(1 for d in TEST_DOCUMENTS if d['true_language'] == 'AR')} Arabic documents")

if __name__ == "__main__":
    generate_test_files()