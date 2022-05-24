from visualize_events.snomed.visualize import *
from visualize_events.snomed.dag import *
from test_dag import predictions_dag, diamond_dag, diamond_dag_tail

T = predictions_dag().compact_preds()

draw_dag(T)
