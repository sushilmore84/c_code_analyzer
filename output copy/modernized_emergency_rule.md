======================================================================
BUSINESS RULE: Emergency Rule
======================================================================

Rule ID: BR-EMERGENCY
Category: CLASSIFICATION
Domains: claim_validation, network, pricing
Execution Phase: Edit, Release
Criticality: HIGH
Type: flag-driven

📋 Description:
  When a claim is for an emergency procedure at an Emergency Room facility, the system marks it as emergency and applies premium pricing.

🔍 IF (Conditions):
  1. Claim.facility_type equals Emergency Room
     Source: validate_emergency_facility
     Relationship: OR
  2. NOT verify_emergency_procedure
     Source: validate_claim
     Relationship: OR

✓ THEN (Actions):
  1. Mark claim as emergency
  2. Apply emergency premium adjustment to base rate

⚡ Triggered By:
  • Validate Emergency Facility
  • Validate Claim

🎯 Used By:
  • Print Summary
  • Validation: Validate Claim
  • Validation: Verify Provider Specialty
  • Pricing: Apply Emergency Premium

💰 Impacts:
  • Patient financial responsibility
  • Claim reimbursement amount
  • Claim processing status

📥 Data Inputs:
  • c->procedure_code
  • c->facility_type

📤 Data Outputs:
  • c->flags
  • c->status
  • c->cos
  • c->patient_responsibility

🔧 Technical Details:
  Flag: FLAG_EMERGENCY
  Implementation Files: network_verification.c, claim_validation.c, main.c, cos_calculation.c
  Setter Functions: 2
  Checker Functions: 4

======================================================================