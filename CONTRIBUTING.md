# Contributing to Pneumonia Chest X-Ray Classification

Thank you for your interest in contributing! We welcome contributions to improve model accuracy, optimize inference speed, extend documentation, and improve the user interface.

## How to Contribute

### 1. Reporting Bugs & Asking Questions
- Open a GitHub issue detailing the bug with steps to reproduce, OS environment, and error tracebacks.
- For model or performance questions, please mention the dataset split and hardware specs used.

### 2. Suggesting Enhancements
- Feature requests (e.g., Grad-CAM explainability, multi-class classification, DICOM image support) are welcomed via GitHub Discussions or Issues.

### 3. Pull Request Guidelines
1. Fork the repository.
2. Create a descriptive feature branch:
   ```bash
   git checkout -b feature/grad-cam-visualization
   ```
3. Make your changes and test them locally:
   ```bash
   python -m py_compile app.py predict.py
   python predict.py --image <sample-image-path>
   ```
4. Commit your changes following conventional commit style:
   ```bash
   git commit -m "feat: add Grad-CAM heatmap overlay to web app"
   ```
5. Push to your fork and submit a Pull Request to `main`.

## Code Style & Standards
- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions.
- Document any new functions or classes with clean docstrings.
- Ensure large dataset files (`train/`, `val/`, `test/`) are never committed to git.
