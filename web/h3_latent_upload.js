import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

app.registerExtension({
  name: "h3.latent.upload.workflow",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!["H3LatentUploadPath", "H3OptionalLatentUploadLoader"].includes(nodeData.name)) return;

    const originalCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalCreated?.apply(this, arguments);
      const node = this;

      node.addWidget("button", "Upload previous H3 latent", null, async () => {
        const picker = document.createElement("input");
        picker.type = "file";
        picker.accept = ".safetensors,application/octet-stream";
        picker.style.display = "none";

        picker.onchange = async () => {
          const file = picker.files?.[0];
          if (!file) return;
          if (!file.name.toLowerCase().endsWith(".safetensors")) {
            alert("Please choose an H3 .safetensors latent file.");
            return;
          }

          const form = new FormData();
          form.append("latent", file, file.name);

          const response = await api.fetchApi("/h3_latent_upload/upload", {
            method: "POST",
            body: form,
          });
          if (!response.ok) {
            throw new Error(`H3 latent upload failed: ${response.status}`);
          }

          const uploaded = await response.json();
          const relative = uploaded.relative_path;
          const combo = node.widgets?.find((widget) => widget.name === "latent_file");
          if (combo) {
            combo.options.values ??= [];
            if (!combo.options.values.includes(relative)) {
              combo.options.values.push(relative);
            }
            combo.value = relative;
            combo.callback?.(relative);
          }
          node.setDirtyCanvas(true, true);
        };

        document.body.appendChild(picker);
        picker.click();
        picker.remove();
      });

      return result;
    };
  },
});
