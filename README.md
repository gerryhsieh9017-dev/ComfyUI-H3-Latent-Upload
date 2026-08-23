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
