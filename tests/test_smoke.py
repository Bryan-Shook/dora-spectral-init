from dora_spectral.smoke import run


def test_all_controlled_conditions_smoke():
    result = run(
        {
            "seed": 0,
            "input_features": 12,
            "output_features": 8,
            "rank": 3,
            "gamma": 0.5,
            "steps": 10,
            "learning_rate": 0.001,
        },
        "cpu",
    )
    assert len(result["conditions"]) == 6
    assert all(row["final_loss"] < row["initial_loss"] for row in result["conditions"])
