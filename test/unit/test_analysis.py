import pandas as pd
import pvinspect as pv
from pvinspect import analysis
from pvinspect.datasets import elpv
from sklearn.metrics import classification_report


def test_defect_classification():
    model = analysis.factory_models.defects()
    data = elpv().pandas.query("testset == True")[:10]
    res = model.apply(data)
    report = pd.DataFrame(
        classification_report(
            res.meta[["crack", "inactive"]],
            res.meta[["pred_crack_p", "pred_inactive_p"]] > 0.5,
            target_names=["crack", "inactive"],
            output_dict=True,
            zero_division=0,
        )
    )

    assert report.loc["f1-score"]["macro avg"] > 0.5
