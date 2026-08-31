from collections import Counter

from onboard.tracker_trial import recommend_trial, resolve_trial_window, summarize_tracks


def test_trial_recommends_candidate_only_with_constrained_gain():
    baseline = summarize_tracks("baseline", "a.yaml", [10] * 100, Counter({i: 20 for i in range(20)}), 10)
    candidate = summarize_tracks("candidate", "b.yaml", [10] * 100, Counter({i: 30 for i in range(15)}), 10)
    assert recommend_trial(baseline, candidate)[0] == "candidate"
    bad = summarize_tracks("candidate", "c.yaml", [8] * 100, Counter({i: 30 for i in range(10)}), 10)
    assert recommend_trial(baseline, bad)[0] == "baseline"


def test_short_video_uses_last_available_window():
    assert resolve_trial_window(360, 300, 120) == (240, 120)
    assert resolve_trial_window(80, 300, 120) == (0, 80)

