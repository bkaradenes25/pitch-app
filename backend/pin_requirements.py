"""
pin_requirements.py -- pin the serving dependencies to the versions installed on THIS machine.

    python pin_requirements.py

model.pkl is a pickle, so the container should use the same xgboost / scikit-learn / numpy / pandas
versions you trained with. Writes requirements-serve.txt (commit it).
"""
from importlib import metadata

PACKAGES = {  # distribution name -> name to write
    "fastapi": "fastapi", "uvicorn": "uvicorn[standard]", "pydantic": "pydantic", "pandas": "pandas",
    "numpy": "numpy", "scikit-learn": "scikit-learn", "xgboost": "xgboost", "joblib": "joblib",
    "requests": "requests", "pyarrow": "pyarrow", "scipy": "scipy",
}


def main():
    lines = []
    for dist, name in PACKAGES.items():
        try:
            lines.append(f"{name}=={metadata.version(dist)}")
        except metadata.PackageNotFoundError:
            lines.append(name)
            print("not installed here, left unpinned:", dist)
    with open("requirements-serve.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("Wrote requirements-serve.txt:\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
