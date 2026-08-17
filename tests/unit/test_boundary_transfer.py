from scripts.evaluate_boundary_transfer import align_transfer_positions


def test_align_transfer_positions_preserves_target_counts_and_source_priorities():
    source = [[0, 4, 8], [0, 8]]
    target = [[0, 3, 6, 9], [0, 6]]
    token_ids = [254, 65, 66, 67, 68, 69, 70, 71, 72, 73]

    transferred = align_transfer_positions(source, target, token_ids)

    assert len(transferred[0]) == len(target[0])
    assert len(transferred[1]) == len(target[1])
    assert {0, 4, 8}.issubset(transferred[0])
    assert transferred[1] == [0, transferred[0].index(8)]
