from dataclasses import dataclass


@dataclass
class Component:
    """Елемент схеми з назвою та вузлами підключення."""
    name: str
    nodes: tuple[str, str]
    value: str


def to_spice(components: list[Component]) -> str:
    # Серіалізація у SPICE-нетліст
    return "\n".join(f"{c.name} {c.nodes[0]} {c.nodes[1]} {c.value}" for c in components)


print(to_spice([Component("R1", ("in", "out"), "1k"), Component("C1", ("out", "0"), "10n")]))
