def get_feedback():
    """
    Ask the user whether the AI response was helpful.
    """

    while True:
        feedback = input(
            "\n👍 Was this response helpful? (yes/no/skip): "
        ).strip().lower()

        if feedback in ["yes", "y"]:
            return "positive"

        elif feedback in ["no", "n"]:
            return "negative"

        elif feedback in ["skip", "s"]:
            return "skip"

        else:
            print("Please enter: yes, no, or skip.")


def get_improvement():
    """
    Ask the user what the AI should learn or improve.
    """

    improvement = input(
        "\n📝 What should EvolveAI improve or learn? "
    ).strip()

    return improvement