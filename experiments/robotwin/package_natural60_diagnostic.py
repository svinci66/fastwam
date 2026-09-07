#!/usr/bin/env python3
"""Archive compact diagnostic evidence and render a static scientific figure."""
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.robotwin.diagnose_natural60_reward import DEFAULT_OUT, PREFIX, sha, write_json


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    archive = ROOT/"experiments/robotwin/results/natural60_20260907"
    archive.mkdir(parents=True,exist_ok=True)
    inventory = json.loads((DEFAULT_OUT/"inventory.json").read_text())
    assert sha(ROOT/"experiments/robotwin/ROBOTWIN_NATURAL60_DIAGNOSTIC_PROTOCOL_20260907.md")==inventory["protocol_sha256"]
    files = ["reward_diagnostic.json","awr_transmission.json","episode_scores.json","prefix_cross_scores.json"]
    for name in files:
        shutil.copyfile(DEFAULT_OUT/name,archive/name)
    write_json(archive/"provenance.json", {
        "schema":"natural60_compact_archive_v1", "development_only":True,
        "protocol_sha256":inventory["protocol_sha256"], "inventory_sha256":sha(DEFAULT_OUT/"inventory.json"),
        "vae_sha256":inventory["vae_sha256"], "source_manifest_sha256":inventory["source_manifest_sha256"],
        "feature_cache_provenance":json.loads((DEFAULT_OUT/"head_latents/provenance.json").read_text()),
        "episodes":inventory["episodes"], "records":len(inventory["records"]),"unique_images":len(inventory["images"]),
        "artifacts":{name:{"local_path":str(DEFAULT_OUT/name),"sha256":sha(DEFAULT_OUT/name)} for name in files+["inventory.json","chunk_scores.json","awr_transition_index.json","awr_transmission_arrays.npz"]},
        "analysis_sources":{name:sha(ROOT/"experiments/robotwin"/name) for name in ["diagnose_natural60_reward.py","diagnose_awr_transmission.py","package_natural60_diagnostic.py"]},
    })
    r = json.loads((archive/"reward_diagnostic.json").read_text())
    fig, axes = plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    colors = {"matched":"#2563eb","donor":"#e07a24","latent_motion":"#677487"}
    labels = {"matched":"Correct imagination","donor":"Other-episode imagination","latent_motion":"Actual motion only"}
    tasks = list(PREFIX)
    x = np.arange(3)
    for offset,key in enumerate(colors):
        values = [r["prefix"][t][key]["estimate"] for t in tasks]
        axes[0].bar(x+(offset-1)*.24,values,width=.23,label=labels[key],color=colors[key])
    axes[0].axhline(.5,color="#777777",ls="--",lw=1)
    axes[0].set(xticks=x,xticklabels=["Microwave\n3 S / 17 F","Mug\n5 S / 15 F","Basket\n10 S / 10 F"],ylim=(0,1.05),ylabel="Success/failure AUC",title="Fixed pre-terminal prefixes")
    axes[0].legend(frameon=False,fontsize=8,loc="lower left")
    keys = ["matched","donor","latent_motion","content_delta"]
    for i,k in enumerate(keys):
        d=r["prefix"]["macro"][k];v=d["estimate"];lo,hi=d["ci95"]
        axes[1].errorbar(v,3-i,xerr=[[v-lo],[hi-v]],fmt="o",color=colors.get(k,"#7c3aed"),capsize=4)
        axes[1].text(hi+.025,3-i,f"{v:.3f}",va="center",fontsize=9)
    axes[1].axvline(0,color="#777777",ls=":",lw=1)
    axes[1].axvline(.5,color="#777777",ls="--",lw=1)
    axes[1].set(yticks=[3,2,1,0],yticklabels=["Correct AUC","Other-episode AUC","Motion-only AUC","Correct - other AUC"],xlim=(-.2,1.06),ylim=(-.6,3.6),title="Task-equal estimates and 95% CIs",xlabel="10,000 episode bootstrap replicates")
    fig.suptitle("Natural-60 development diagnostic: useful correlation, unproven content increment",fontsize=12)
    fig.savefig(archive/"reward_diagnostic.png",dpi=180)
    fig.savefig(DEFAULT_OUT/"reward_diagnostic.pdf")
    plt.close(fig)
    print(archive)


if __name__=="__main__":main()
