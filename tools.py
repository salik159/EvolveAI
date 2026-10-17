import math


def calculator(expression):
    """
    Safely evaluates basic mathematical expressions.
    """

    allowed_names = {
        "abs": abs,
        "round": round,
        "sqrt": math.sqrt,
        "pow": pow,
        "pi": math.pi,
    }

    try:
        result = eval(
            expression,
            {"__builtins__": {}},
            allowed_names
        )

        return str(result)

    except Exception as error:
        return f"Calculator error: {error}"