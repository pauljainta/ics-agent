import json

from scripts.plcbench_pilot_crosswalk import crosswalk_from_rescored


def test_crosswalk_keeps_objectives_and_missing_physical_evidence_separate(tmp_path):
    cases = []
    for index, objective in enumerate(("versioned:pilot_wd_loss_v1.json",
                                       "versioned:pilot_wt_loss_v1.json")):
        bundle = tmp_path / f"case-{index}"
        bundle.mkdir()
        (bundle / "meta.json").write_text(json.dumps({
            "workload": "water_distribution" if index == 0 else "water_treatment",
            "arm": "native"}))
        cases.append({"bundle": str(bundle), "episode_id": f"case-{index}",
                      "status": "assessable", "objective_source": objective,
                      "matches_original": True,
                      "task_specific": {"flags": {"discover": True, "read": True,
                                                  "write": False, "manipulate": False,
                                                  "disrupt": False, "impact": False},
                                        "physical": {"disrupt": {"assessable": True},
                                                     "impact": {"assessable": index == 0}}}})

    report = crosswalk_from_rescored({"cases": cases}, tmp_path)
    assert report["totals"] == {"episodes": 2, "assessable": 2,
                                "unassessable": 0, "matching_original": 2}
    assert len(report["groups"]) == 2
    assert report["pilot_completeness"] == "incomplete"
    assert all(group["attempted"] == 1 for group in report["groups"])
    wt = next(group for group in report["groups"] if "wt_loss" in group["objective_source"])
    assert wt["flag_assessable"]["impact"] == 0
    assert next(case for case in report["cases"] if case["bundle"] == "case-1")["flags"]["impact"] is None
    assert all(item["status"] == "unassessable"
               for item in report["ics_agent_metric_availability"].values())
