from darkforest.archetypes import ARCHETYPES


def test_archetypes_expose_policy_interface():
    for name, cls in ARCHETYPES.items():
        assert hasattr(cls, "init_carry"), name
        assert hasattr(cls, "act"), name
