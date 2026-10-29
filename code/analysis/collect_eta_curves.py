"""Convert named ETA --keep_curves outputs to the waveform-figure input schema."""
import argparse
import json
from validation import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, metavar="LABEL=JSON")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = {}
    for item in args.input:
        label, path = item.split("=", 1)
        if label in result:
            raise ValueError(f"Duplicate figure label: {label}")
        with open(path) as handle:
            row = json.load(handle)["osz"]
        event, control = row["curve_event"], row["curve_control"]
        if len(event) != 61 or len(control) != 61:
            raise ValueError("ETA figure requires two61-point curves at10ms frames")
        result[label] = {"event": event, "control": control}
    save_json(args.out, result)


if __name__ == "__main__":
    main()
