import json
import argparse
from pathlib import Path

REPORT_TEMPLATE = Path(__file__).parent.parent / "aso_workflow" / "report_template.html"

def write_report_html(json_path: str, html_path: str) -> str:
    """Write the report to an HTML file."""
    with open(json_path, "r") as f:
        report = json.load(f)
    with open(REPORT_TEMPLATE, "r") as f:
        template = f.read()
    template = template.replace("const REPORT = {}", f"const REPORT = {json.dumps(report, indent=2)}")
    with open(html_path, "w") as f:
        f.write(template)
    print(f"Saved report to {html_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in-file", type=str, help="Path to json report from ASO Assessment Pipeline")
    args = parser.parse_args()

    assert args.in_file.endswith(".json")

    json_path, html_path = args.in_file, args.in_file.replace(".json", ".html")
    write_report_html(json_path, html_path)