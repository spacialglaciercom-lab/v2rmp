🧹 [Fix unused local Config class in re_export_onnx.py]

🎯 **What:** The local duplicated `@dataclass class Config` definition in `re_export_onnx.py` was removed and replaced with an import `from train_job import Config`. Also cleaned up an unused `from dataclasses import dataclass` import.

💡 **Why:** The codebase AST analysis identified the local `Config` class as defined but never instantiated locally. By importing the class from `train_job.py` (which the comment explicitly says it must match exactly), we resolve the duplication, ensuring both files stay in sync and improving code maintainability.

✅ **Verification:**
1. Ran python script tests demonstrating that `torch.load` successfully unpickles the checkpoint weights using the imported `Config` class.
2. Verified that the overall rust codebase (`cargo test`) was not negatively affected by this pure-python script cleanup (the pre-existing rust compilation errors remain untouched).
3. Verified the unused dataclass import was properly removed.

✨ **Result:** A cleaner, DRY (Don't Repeat Yourself) script that securely references the correct `Config` class required for ONNX model export.
