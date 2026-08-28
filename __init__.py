import os
import re

import folder_paths
from aiohttp import web
from server import PromptServer

try:
    from safetensors.torch import load_file as _st_load
except Exception:
    _st_load = None


_UPLOAD_SUBFOLDER = "h3_latents"
_UPLOAD_REQUIRED = "(upload required)"


def _safe_latent_name(filename):
    name = os.path.basename(filename or "previous_h3_latent.safetensors")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    if not name.lower().endswith(".safetensors"):
        raise ValueError("H3 motion-context latent must be a .safetensors file.")
    return name


@PromptServer.instance.routes.post("/h3_latent_upload/upload")
async def upload_h3_latent(request):
    reader = await request.multipart()
    field = await reader.next()
    if field is None or field.name != "latent":
        return web.json_response({"error": "Missing latent upload."}, status=400)

    try:
        filename = _safe_latent_name(field.filename)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)

    input_dir = folder_paths.get_input_directory()
    target_dir = os.path.join(input_dir, _UPLOAD_SUBFOLDER)
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, filename)
    temporary_path = target_path + ".uploading"

    try:
        with open(temporary_path, "wb") as handle:
            while True:
                chunk = await field.read_chunk(size=1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        os.replace(temporary_path, target_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)

    return web.json_response({
        "name": filename,
        "subfolder": _UPLOAD_SUBFOLDER,
        "relative_path": f"{_UPLOAD_SUBFOLDER}/{filename}",
    })


class H3LatentUploadPath:
    """Workflow-side uploader/path bridge for H3 motion-context safetensors."""

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = []
        for root, _, names in os.walk(input_dir):
            for name in names:
                if name.lower().endswith(".safetensors"):
                    full_path = os.path.join(root, name)
                    files.append(os.path.relpath(full_path, input_dir).replace(os.sep, "/"))
        return {
            "required": {
                "latent_file": ([_UPLOAD_REQUIRED, *sorted(files)],),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("latent_path",)
    FUNCTION = "resolve"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Upload or select an H3 motion-context .safetensors file inside the "
        "Workflow editor, then pass its absolute path to H3 Motion Context Load Latent."
    )

    def resolve(self, latent_file):
        if latent_file == _UPLOAD_REQUIRED:
            raise FileNotFoundError("Please upload the previous H3 latent file.")
        path = folder_paths.get_annotated_filepath(latent_file)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"H3 latent file not found: {latent_file}")
        return (path,)

    @classmethod
    def VALIDATE_INPUTS(cls, latent_file):
        if not latent_file or latent_file == _UPLOAD_REQUIRED:
            return "Please upload or select the previous H3 latent file."
        path = folder_paths.get_annotated_filepath(latent_file)
        if not os.path.isfile(path):
            return f"H3 latent file not found: {latent_file}"
        if not latent_file.lower().endswith(".safetensors"):
            return "H3 motion-context latent must be a .safetensors file."
        return True


class H3OptionalLatentUploadLoader:
    """Load an uploaded H3 context latent only when explicitly enabled.

    Returning ``None`` while disabled is intentional: H3 Motion Context marks
    ``context_latent`` as optional and then falls back to the connected MP4
    frames/audio.  Keeping the enable check inside this node avoids ComfyUI
    evaluating an unused file-loader branch before a downstream switch.
    """

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = []
        for root, _, names in os.walk(input_dir):
            for name in names:
                if name.lower().endswith(".safetensors"):
                    full_path = os.path.join(root, name)
                    files.append(
                        os.path.relpath(full_path, input_dir).replace(os.sep, "/")
                    )
        return {
            "required": {
                "use_context_latent": ("BOOLEAN", {"default": False}),
                "latent_file": ([_UPLOAD_REQUIRED, *sorted(files)],),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("optional_context_latent",)
    FUNCTION = "load_optional"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "When disabled, returns no Context Latent and lets H3 Motion Context "
        "use the previous MP4 frames/audio. When enabled, loads the uploaded "
        "H3 video+audio .safetensors file."
    )

    def load_optional(self, use_context_latent, latent_file):
        if not use_context_latent:
            return (None,)
        if _st_load is None:
            raise RuntimeError(
                "safetensors is unavailable; cannot load the H3 Context Latent."
            )
        if not latent_file or latent_file == _UPLOAD_REQUIRED:
            raise FileNotFoundError(
                "Context Latent is enabled. Upload the previous H3 .safetensors file."
            )
        path = folder_paths.get_annotated_filepath(latent_file)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"H3 Context Latent not found: {latent_file}")
        data = _st_load(path)
        if "video" not in data or "audio" not in data:
            raise ValueError(
                "The selected file is not an H3 Motion Context latent "
                "(video/audio tensors are required)."
            )
        return ({"samples": [data["video"], data["audio"]]},)

    @classmethod
    def VALIDATE_INPUTS(cls, use_context_latent, latent_file):
        if not use_context_latent:
            return True
        if not latent_file or latent_file == _UPLOAD_REQUIRED:
            return (
                "Context Latent is enabled. Upload the previous H3 "
                ".safetensors file."
            )
        path = folder_paths.get_annotated_filepath(latent_file)
        if not os.path.isfile(path):
            return f"H3 Context Latent not found: {latent_file}"
        if not latent_file.lower().endswith(".safetensors"):
            return "H3 Motion Context latent must be a .safetensors file."
        return True

    @classmethod
    def IS_CHANGED(cls, use_context_latent, latent_file):
        if not use_context_latent:
            return "disabled"
        if not latent_file or latent_file == _UPLOAD_REQUIRED:
            return float("NaN")
        try:
            path = folder_paths.get_annotated_filepath(latent_file)
            return f"{path}:{os.stat(path).st_mtime_ns}"
        except Exception:
            return float("NaN")


NODE_CLASS_MAPPINGS = {
    "H3LatentUploadPath": H3LatentUploadPath,
    "H3OptionalLatentUploadLoader": H3OptionalLatentUploadLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3LatentUploadPath": "H3 Latent Upload (Workflow)",
    "H3OptionalLatentUploadLoader": "H3 Optional Context Latent Upload + Load",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
