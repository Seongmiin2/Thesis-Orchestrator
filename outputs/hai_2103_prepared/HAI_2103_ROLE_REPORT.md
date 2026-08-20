# HAI 21.03 Point-Role Gate

## Overall Assessment: PASS_TO_MODELING

Roles were assigned manually from the official HAI technical manual, pages 12-14. Version-specific aliases whose semantics are not printed verbatim in the latest table are marked `medium` confidence.

All `79` SCADA points are assigned exactly once: `29` sensor/model points and `50` control-history points.
After removing train-constant points, F0 has `29` inputs and F1 has `57` inputs (`28` added control-history channels).
Mapping confidence: `{'high': 64, 'medium': 15}`.

F0 includes physical measurements and HIL model signals. F1 adds setpoints, controller outputs, commands, actuator states, and operating modes. Constant training channels are excluded from both models before scaling.

| process   | role           |   points |
|:----------|:---------------|---------:|
| P1        | ACTUATOR_STATE |        9 |
| P1        | CO             |       10 |
| P1        | MODE           |        1 |
| P1        | MODEL_SIGNAL   |        3 |
| P1        | PV             |        8 |
| P1        | SP             |        7 |
| P2        | CO             |        3 |
| P2        | MODE           |        3 |
| P2        | MODEL_SIGNAL   |        1 |
| P2        | PV             |        8 |
| P2        | SP             |        7 |
| P3        | CO             |        2 |
| P3        | PV             |        3 |
| P3        | SP             |        2 |
| P4        | CO             |        1 |
| P4        | MODEL_SIGNAL   |        6 |
| P4        | SP             |        5 |
