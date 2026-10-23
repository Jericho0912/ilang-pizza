"""``python -m ilang_pizza``: same as the ``ilang-pizza`` command."""

from ilang_pizza.cli import app

if __name__ == "__main__":
    app(prog_name="ilang-pizza")
