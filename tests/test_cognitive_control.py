from cognition import ExecutiveController


def main():
    controller = ExecutiveController()
    examples = [
        "Hola, ¿cómo estás?",
        "¿Qué recuerdas sobre Aster?",
        "Con toda la información que te he dado anteriormente, ¿alcanzan las reservas?",
        "Revisa lo que dijiste antes y corrige cualquier contradicción.",
        "Calcula el 25% de 40000.",
        "Diseña un plan para ampliar la colonia.",
    ]

    for query in examples:
        state = controller.analyze(query)
        print("=" * 80)
        print("QUERY:", query)
        print("INTENT:", state.intent)
        print("SECONDARY:", state.secondary_intents)
        print("TOOLS:", state.tools_needed)
        print("CONFIDENCE:", state.decision_confidence)


if __name__ == "__main__":
    main()
