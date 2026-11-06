"""Export scanner/radiotracer labels from NASC DICOM headers to a manifest."""
import argparse
import csv
from pathlib import Path
import pydicom


def domain_from_dicom(path):
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    manufacturer = str(ds.get("Manufacturer", "")).upper()
    if "SIEMENS" in manufacturer:
        scanner = "Siemens"
    elif "SINO" in manufacturer:
        scanner = "SinoUnion"
    elif "UIH" in manufacturer or "UNITED IMAGING" in manufacturer:
        scanner = "UIH"
    else:
        raise ValueError("Unrecognized scanner manufacturer: " + manufacturer)
    sequence = ds.get("RadiopharmaceuticalInformationSequence", [])
    descriptions = [str(item.get("Radiopharmaceutical", "")).upper() for item in sequence]
    text = " ".join(descriptions)
    candidates = [name for name in ("DOTATATE", "CXCR4", "PSMA", "MFBG", "FAPI", "DOPA", "FDG") if name in text]
    if len(candidates) != 1:
        raise ValueError("DICOM radiotracer is missing/ambiguous; supply verified metadata explicitly")
    return scanner, candidates[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV with dicom_path and de-identified patient/study/split/NASC/ASC fields")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.input).resolve()
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        dicom = (source.parent / row["dicom_path"]).resolve()
        row["scanner"], row["radiotracer"] = domain_from_dicom(dicom)
        for key in ("nasc", "asc"):
            if row.get(key):
                row[key] = str((source.parent / row[key]).resolve())
        del row["dicom_path"]
    with Path(args.output).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
