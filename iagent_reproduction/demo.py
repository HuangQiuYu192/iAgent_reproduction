"""A small, observable iAgent/i²Agent walkthrough."""

from .backends import TeachingBackend
from .core import I2Agent, IAgent, Interaction, Item


def main() -> None:
    history = [Interaction("u1", Item("h1", "Quiet Cosmos", "a philosophical science fiction novel"), 1,
                           "I enjoyed the thoughtful moral questions."),
               Interaction("u1", Item("h2", "Alien Contact", "cerebral story of first contact"), 2,
                           "Great reflective science fiction.")]
    slate = [Item("a", "Deep Space Ethics", "a cerebral philosophical science fiction journey"),
             Item("b", "Quick Baking", "weeknight dessert recipes")]
    instruction = "I want a thoughtful science fiction story with philosophical themes."
    backend = TeachingBackend()
    basic, dynamic = IAgent(backend), I2Agent(backend)
    print("iAgent:", basic.rank(history, instruction, slate))
    dynamic.learn_feedback(history, slate[1])
    print("I2Agent profile:", dynamic.profile)
    print("I2Agent:", dynamic.rank(history, instruction, slate))


if __name__ == "__main__":
    main()
