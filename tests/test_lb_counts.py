"""Целые числа по типам данных из лога площадки (scripts/lb_counts.py)."""
import importlib.util
from pathlib import Path

LOG = """final_score:        0.3285007452
Время инференса:    3 мин 11 с / 30 мин
  Soft recall:        0.426213
  Center deviation:   1.478164 м
  FP ratio:           1.548218
По типам данных:
  лидар      precision 0.508658   recall 0.487552
  аэрофото   precision 0.090909   recall 0.013333
  спутник    precision 0.384615   recall 0.291777
  ортофото   precision 0.000000   recall 0.000000
  геофизика  precision 0.791092   recall 0.258848
"""


def _mod():
    path = Path(__file__).resolve().parents[1] / "scripts" / "lb_counts.py"
    spec = importlib.util.spec_from_file_location("lb_counts", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_counts_of_the_24_10_log():
    m = _mod()
    parsed = m.parse(LOG)
    assert parsed["final_score"] == 0.3285007452 and parsed["groups"]["ортофото"] == (0.0, 0.0)
    sol = m.resolve(parsed)[0]
    assert sol == {"лидар": (470, 908, 964), "аэрофото": (1, 20, 75), "спутник": (110, 352, 377),
                   "геофизика": (373, 197, 1441)}
    assert m.hidden_gt(parsed, sol)[0] == (50, 1239)


def test_edge_is_f_over_four_minus_f():
    m = _mod()
    f1, hit, e = m.edge(373, 197, 1441)
    assert abs(f1 - 2 * 373 / (1441 + 373 + 98.5)) < 1e-12
    assert abs(e - f1 / (4 - f1)) < 1e-12 and abs(hit - 373 / 570) < 1e-12
