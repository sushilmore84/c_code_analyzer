# Emergency_Classification

**Description:** Determines if a claim should be marked as emergency based on facility type or failed emergency procedure verification, triggering premium pricing adjustments
**Hit Policy:** FIRST

| Rule # | Facility Type | Emergency Procedure Valid | Emergency Classification |
|---|---|---|---|
| 1 | Emergency Room | Yes | Emergency |
| 2 | Emergency Room | No | Emergency |
| 3 | Non-Emergency Room | No | Emergency |
| 4 | Non-Emergency Room | Yes | Not Emergency |

## Input Expressions

- **Facility Type**: `claim.facility_type`
- **Emergency Procedure Valid**: `verify_emergency_procedure(claim)`