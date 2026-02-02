# Labeling Guide — Starship Incident Dataset

This guide defines label fields and consistency rules for annotating Starship incident narratives.

## General principles

- **Evidence only:** assign labels based solely on the incident narrative text and cited sources.
- **No memory-based labeling:** do not label from memory or outside knowledge.
- **Uncertainty:** if the narrative speculates or is ambiguous, use `cause=hypothesis_only` or leave the cause empty.
- **Minimalism:** select only labels supported by the text.

## Label fields

### Subsystem
Indicate which subsystem is most directly implicated.

- `propulsion`: engines, propellants, combustion, thrust, ignition.
- `structures`: tanks, body, structural integrity, collapse.
- `avionics`: sensors, flight computers, electronics.
- `guidance_navigation_control`: attitude control, navigation, guidance software.
- `thermal_protection`: heat shield, thermal tiles, reentry heating.
- `ground_support`: pad equipment, tank farm, launch mount.
- `flight_termination`: autonomous termination system or manual termination.
- `software`: control software faults not strictly tied to GNC hardware.
- `communications`: telemetry or communications links.
- `power`: power systems or batteries.
- `recovery`: landing hardware, recovery ops, splashdown systems.
- `operations`: procedural or operational issues.
- `unspecified`: use only when none of the above are supported by text.

### Failure mode

- `engine_failure`: engine did not perform or shut down unexpectedly.
- `pressurization_failure`: tank pressurization issues during tests.
- `structural_failure`: collapse or structural break.
- `fire`: fire without confirmed explosion.
- `explosion`: rapid destructive event or RUD.
- `loss_of_control`: guidance/attitude instability or control loss.
- `separation_failure`: failure of stage separation.
- `landing_failure`: landing or recovery failure.
- `ignition_abort`: ignition sequence stopped before firing.
- `telemetry_loss`: loss of telemetry or comms link.
- `pad_damage`: damage to launch infrastructure.
- `debris_impact`: damage from debris.
- `thermal_damage`: thermal protection issues.
- `propellant_leak`: leak during tanking/ops.
- `valve_failure`: valve malfunction.
- `sensor_failure`: sensor fault in checks.
- `abort`: generic abort without clear mode.

### Impact

- `vehicle_loss`: loss of the combined vehicle.
- `booster_loss`: booster-only loss.
- `ship_loss`: ship-only loss.
- `pad_damage`: damage to ground infrastructure.
- `test_abort`: test ended before completion.
- `mission_loss`: mission objectives not achieved.
- `partial_damage`: damage without total loss.
- `safe_abort`: safe termination without loss.
- `successful_recovery`: controlled landing/recovery.

### Cause

- `unknown`: no evidence for specific cause.
- `design_issue`: evidence points to design limitations.
- `manufacturing_defect`: evidence points to manufacturing quality.
- `operational_error`: procedural or human error.
- `environmental`: weather or external conditions.
- `hypothesis_only`: speculation without evidence.

## Examples

- **Text:** "Telemetry dropped during ascent; flight was terminated." → `communications`, `flight_termination`; `telemetry_loss`; `mission_loss`.
- **Text:** "Hard landing and vehicle exploded on the pad." → `guidance_navigation_control`, `propulsion`; `landing_failure`, `explosion`; `vehicle_loss`, `pad_damage`.

## Notes field

Use notes to justify uncertain labels, cite ambiguous phrases, or record follow-up needed for verification.
