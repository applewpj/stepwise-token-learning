from loss_acquisition.synthetic_mano._mano import generate_sample, length_weights


def test_mano_generator_matches_physicslm_golden_seeds():
    weights = length_weights(explicit={i: 0.1 for i in range(1, 11)})
    assert generate_sample(0, weights).tokens == [
        9998,
        90,
        5,
        1,
        1,
        5019,
        2,
        7,
        5009,
        1,
        5021,
        5010,
        5011,
        7,
        1,
        6,
        5001,
        5017,
        5002,
        5022,
        5010,
        94,
        5012,
    ]
    assert generate_sample(1_000_000_000_000, weights).tokens == [
        9998,
        50,
        3,
        7,
        7,
        5,
        2,
        5013,
        5011,
        5005,
        5010,
        5014,
        5021,
        55,
        5018,
    ]
