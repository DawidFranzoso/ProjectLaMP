import json
import matplotlib.pyplot as plt


def draw_output(name: str):
    with open(f"vis_outputs/{name}") as f:
        data = json.load(f)

    for prefix in ("", "val_"):
        plt.suptitle(f"{name}\n{prefix}similarities over {len(data)} epochs")
        plt.hlines(data[-1].get("triplet_alpha", 0.9), xmin=0, xmax=len(data) - 1, label="triplet loss epsilon", color="orange")
        plt.plot([d[f"{prefix}negative_similarity"] for d in data], c="red", label=f"{prefix}negative similarity")
        plt.plot([d[f"{prefix}positive_similarity"] for d in data], c="green", label=f"{prefix}positive similarity")
        plt.legend()
        plt.show()

        plt.suptitle(f"{name}\n{prefix}loss over {len(data)} epochs")
        plt.plot([d[f"{prefix}loss"] for d in data], c="blue", label=f"{prefix}loss")
        plt.legend()
        plt.show()


draw_output("similarity_1_weight_5_5e-5")
