# H3 Latent Upload (Workflow)

Adds a Workflow-side `.safetensors` upload button for MiniMax H3 motion-context
latents. The node returns the uploaded file's absolute path for connection to
`MiniMaxH3MotionContextLoadLatent.latent_path`.

Uploaded files are stored in ComfyUI's `input/h3_latents` directory inside the
active session. Uploading another file with the same name replaces that session
copy. No media, prompts, or latent files are included in this repository.

This deliberately does not use ComfyUI's stock `Load Latent`: H3 motion-context
files store separate `video` and `audio` tensors and must be decoded by the
matching H3 Motion Context loader.

## Optional upload + load node

`H3 Optional Context Latent Upload + Load` is the safe node for a workflow that
must support both MP4-only continuation and optional lossless Context Latent
continuation in one branch.

- `use_context_latent = false`: no file is required or opened; the node returns
  no optional latent, so H3 Motion Context falls back to the connected previous
  MP4 frames and audio.
- `use_context_latent = true`: upload/select the previous H3 `.safetensors`;
  the node validates and loads its separate `video` and `audio` tensors.

The enable check deliberately lives inside the loader. A normal downstream
switch may still evaluate both upstream branches in ComfyUI, which can otherwise
trigger a disabled file loader or even compute two video-generation branches.

## Safe External LoRA nodes

`H3 External LoRA Settings (Safe)` exposes a direct `.safetensors` URL, a local
cache filename, and an optional bearer token. It only passes strings and never
downloads by itself.

`H3 Safe External LoRA Apply (Model)` combines the enable check, download,
validation, and model patch in one node:

- `enable_lora = false`: returns the original model immediately without network
  access, filesystem access, a LoRA model-list lookup, or an upstream download.
- `enable_lora = true`: downloads to `input/h3_external_loras`, first writing a
  `.part` file and then atomically renaming it. The file is loaded with ComfyUI's
  safe loader and applied directly to the connected model.
- Public Hugging Face repositories need no token. A private repository needs a
  fine-grained/read token in `bearer_token` for the current session.
- Never commit a workflow while a bearer token is present. Clear it after use.
- HTTP 401/403, empty downloads, and invalid `.safetensors` files fail before
  sampling with a specific error and do not leave a partial cache file.

This avoids relying on a Machine's first registered `models/loras` directory or
on a `lora_name` COMBO value being present in its current model index.
