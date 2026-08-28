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
