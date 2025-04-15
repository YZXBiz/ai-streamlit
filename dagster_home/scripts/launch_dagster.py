from clustering.dagster.definitions import defs
if __name__ == "__main__":
    print("Launching Dagster with definitions")
    from dagster import cli
    cli.main(["dev"])
