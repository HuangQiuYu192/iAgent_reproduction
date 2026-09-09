from iagent_reproduction.backends import TeachingBackend
from iagent_reproduction.core import I2Agent, IAgent, Interaction, Item, reflect


def sample():
    history = [Interaction("u", Item("h", "Space", "philosophical science fiction"), 1, "Loved it")]
    candidates = [Item("good", "Planet", "thoughtful science fiction"), Item("bad", "Cake", "baking recipes")]
    return history, candidates


def test_reflection_rejects_hallucinated_ids():
    _, candidates = sample()
    assert reflect(candidates, ["good", "invented"]) == ["good", "bad"]


def test_iagent_returns_permutation():
    history, candidates = sample()
    assert set(IAgent(TeachingBackend()).rank(history, "thoughtful science fiction", candidates)) == {"good", "bad"}


def test_i2agent_updates_only_its_own_profile():
    history, candidates = sample()
    first, second = I2Agent(TeachingBackend()), I2Agent(TeachingBackend())
    first.learn_feedback(history, candidates[1])
    assert first.profile != second.profile
    assert set(first.rank(history, "science fiction", candidates)) == {"good", "bad"}
