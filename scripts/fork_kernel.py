"""Pull a public Kaggle kernel and prepare it as a private fork under kernels/<slug>.

Keeps the author's pinned Docker image (dropping it cost another team 0.04 on the board).
Usage: python scripts/fork_kernel.py <owner/kernel> <our-slug> [--gpu]
"""
import json, os, subprocess, sys, glob, shutil

USER = "bryantyu"
src, ours = sys.argv[1], sys.argv[2]
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = os.path.join(root, "kernels", ours)
shutil.rmtree(d, ignore_errors=True)
os.makedirs(d)
subprocess.run(["kaggle", "kernels", "pull", src, "-p", d, "-m"], check=True, capture_output=True)
nbs = glob.glob(os.path.join(d, "*.ipynb"))
os.rename(nbs[0], os.path.join(d, "notebook.ipynb"))
m = json.load(open(os.path.join(d, "kernel-metadata.json")))
out = {
    "id": f"{USER}/{ours}", "title": ours, "code_file": "notebook.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": m.get("enable_gpu", False),
    "enable_tpu": False, "enable_internet": False,
    "dataset_sources": [x for x in m.get("dataset_sources", []) if x],
    "competition_sources": ["enveda-CASMI26-molecule-id-mass-spectra"],
    "kernel_sources": [x for x in m.get("kernel_sources", []) if x],
    "model_sources": [x for x in m.get("model_sources", []) if x],
}
if m.get("machine_shape"):
    out["machine_shape"] = m["machine_shape"]
if m.get("docker_image"):
    out["docker_image"] = m["docker_image"]
    out["docker_image_pinning_type"] = "original"
json.dump(out, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=2)
print("prepared", out["id"], "datasets", len(out["dataset_sources"]), "gpu", out["enable_gpu"])
