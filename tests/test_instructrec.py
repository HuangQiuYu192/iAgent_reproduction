from iagent_reproduction.instructrec import load_examples


def test_released_books_protocol_is_preserved():
    examples = load_examples("data/instructrec", "books", limit=2)
    assert examples
    example = examples[0]
    assert len(example.candidates) == 10
    assert example.target_id in [item.item_id for item in example.candidates]
    assert example.history and example.instruction
