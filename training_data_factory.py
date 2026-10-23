"""Canonical config-driven training data factory entrypoint."""


def main() -> None:
    from modules.training_data_factory_postgis_runner import main as _main

    _main()


if __name__ == "__main__":
    main()
