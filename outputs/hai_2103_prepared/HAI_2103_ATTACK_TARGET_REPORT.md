# HAI 21.03 Attack-Target Gate

## Overall Assessment: PASS_WITH_SCOPE_LIMITATION

The 50 labeled global attack intervals were matched in chronological order to the official HAI 21.03 attack table on technical-manual pages 36-38. Target aliases were normalized to the recorded 21.03 column names and validated against the point-role table.

Target classes: `{'control_only': 31, 'mixed_control_sensor': 19}`.
Every event directly targets at least one control-history point. HAI 21.03 therefore tests transfer under direct control-target and mixed control-plus-sensor attacks; it cannot independently establish utility for attacks with no directly manipulated control channel.

| file         | target_class         |   events |
|:-------------|:---------------------|---------:|
| test1.csv.gz | control_only         |        5 |
| test2.csv.gz | control_only         |       12 |
| test2.csv.gz | mixed_control_sensor |        8 |
| test3.csv.gz | control_only         |        6 |
| test3.csv.gz | mixed_control_sensor |        2 |
| test4.csv.gz | control_only         |        1 |
| test4.csv.gz | mixed_control_sensor |        4 |
| test5.csv.gz | control_only         |        7 |
| test5.csv.gz | mixed_control_sensor |        5 |
