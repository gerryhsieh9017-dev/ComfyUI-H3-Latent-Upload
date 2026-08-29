import os
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import folder_paths
from aiohttp import web
from server import PromptServer

import comfy.sd
import comfy.utils

try:
    from safetensors.torch import load_file as _st_load
except Exception:
    _st_load = None


_UPLOAD_SUBFOLDER = "h3_latents"
_UPLOAD_REQUIRED = "(upload required)"
_LORA_CACHE_SUBFOLDER = "h3_external_loras"
_HF_TOKEN_ENV = "H3_LORA_HF_TOKEN"


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


def _safe_lora_name(filename):
    name = os.path.basename(filename or "character_lora.safetensors")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    if not name.lower().endswith(".safetensors"):
        raise ValueError("External LoRA save name must end with .safetensors.")
    return name


def _resolve_bearer_token(bearer_token, lora_url):
    """Use the Machine Secret only for Hugging Face download hosts."""

    token = (bearer_token or "").strip()
    if token:
        return token
    hostname = (urlparse(lora_url).hostname or "").lower()
    if hostname == "huggingface.co" or hostname.endswith(".huggingface.co"):
        return os.environ.get(_HF_TOKEN_ENV, "").strip()
    return ""


class H3ExternalLoRASettings:
    """Non-downloading UI bridge for URL, cache name, and optional HF token."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_url": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Direct Hugging Face resolve/main URL.",
                    },
                ),
                "lora_save_name": (
                    "STRING",
                    {"default": "character_lora.safetensors"},
                ),
                "bearer_token": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Private repo: leave blank to use the ComfyDeploy Machine "
                            "Secret H3_LORA_HF_TOKEN. A session-only value entered here "
                            "takes priority; never Commit a workflow containing a token."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("lora_url", "lora_save_name", "bearer_token")
    FUNCTION = "settings"
    CATEGORY = "loaders/minimax"
    DESCRIPTION = (
        "Stores External LoRA settings only. It never downloads a file by itself. "
        "The paired Safe External LoRA Apply node downloads only when Enable is ON."
    )

    def settings(self, lora_url, lora_save_name, bearer_token):
        return (lora_url.strip(), _safe_lora_name(lora_save_name), bearer_token.strip())


class H3SafeExternalLoRAApplyModel:
    """Download and apply an external LoRA only when explicitly enabled.

    The official ComfyDeploy External LoRA node writes to the first registered
    LoRA directory. On some Machines that path is absent even though the actual
    private-model path exists. This node intentionally caches under ComfyUI's
    input directory, creates its own directory, checks HTTP failures, writes
    atomically, validates with ComfyUI's safe loader, and applies the state dict
    directly. No COMBO/model-list refresh is involved.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enable_lora": ("BOOLEAN", {"default": False}),
                "lora_url": ("STRING", {"default": ""}),
                "lora_save_name": (
                    "STRING",
                    {"default": "character_lora.safetensors"},
                ),
                "bearer_token": ("STRING", {"default": ""}),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "loaders/minimax"
    DESCRIPTION = (
        "OFF returns the original model without network or file access. ON downloads "
        "a direct .safetensors URL into input/h3_external_loras and applies it."
    )

    @staticmethod
    def _http_error(status, url=""):
        host = (urlparse(url).hostname or "").lower()
        if status in (401, 403):
            if host == "civitai.red" or host.endswith(".civitai.red"):
                return RuntimeError(
                    "External LoRA download was denied by Civitai (HTTP %s). "
                    "Use the direct /api/download/models/<version_id> URL. If "
                    "the URL is correct, the remote service may be temporarily "
                    "blocking automated downloads." % status
                )
            return RuntimeError(
                "External LoRA download was denied (HTTP %s). For a private "
                "Hugging Face repository, configure the ComfyDeploy Machine Secret "
                "H3_LORA_HF_TOKEN with a fine-grained/read token, or paste a token "
                "into bearer_token for this Session only. Do not Commit the token." % status
            )
        return RuntimeError("External LoRA download failed with HTTP %s." % status)

    def _download(self, url, destination, bearer_token):
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(
                "External LoRA URL must be a direct http(s) .safetensors URL."
            )

        # Some public download services (including Civitai) reject Python's
        # default urllib user agent even though the same public URL works in a
        # browser. Use an explicit, stable client identity and binary Accept.
        headers = {
            "User-Agent": "ComfyUI-H3-Latent-Upload/1.0",
            "Accept": "application/octet-stream,*/*;q=0.8",
        }
        if bearer_token:
            headers["Authorization"] = f"Bearer {bearer_token}"

        temporary = destination + ".part"
        if os.path.exists(temporary):
            os.remove(temporary)
        try:
            request = Request(url, headers=headers)
            try:
                response = urlopen(request, timeout=300)
            except HTTPError as exc:
                raise self._http_error(exc.code, url) from exc
            except URLError as exc:
                raise RuntimeError(
                    f"External LoRA download could not connect: {exc.reason}"
                ) from exc
            with response, open(temporary, "wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            if not os.path.isfile(temporary) or os.path.getsize(temporary) < 16:
                raise ValueError("External LoRA download was empty or incomplete.")
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)

    def _load_or_download(self, lora_url, lora_save_name, bearer_token):
        cache_dir = os.path.join(
            folder_paths.get_input_directory(), _LORA_CACHE_SUBFOLDER
        )
        os.makedirs(cache_dir, exist_ok=True)
        destination = os.path.join(cache_dir, _safe_lora_name(lora_save_name))

        if os.path.isfile(destination):
            try:
                return comfy.utils.load_torch_file(destination, safe_load=True)
            except Exception:
                os.remove(destination)

        self._download(lora_url, destination, bearer_token)
        try:
            return comfy.utils.load_torch_file(destination, safe_load=True)
        except Exception as exc:
            if os.path.exists(destination):
                os.remove(destination)
            raise ValueError(
                "Downloaded file is not a valid safe LoRA .safetensors file. "
                "Check the direct URL, repository access, and LoRA file."
            ) from exc

    def apply(
        self,
        model,
        enable_lora,
        lora_url,
        lora_save_name,
        bearer_token,
        strength_model,
    ):
        if not enable_lora:
            return (model,)
        if not lora_url or not lora_url.strip():
            raise ValueError(
                "External LoRA is enabled, but lora_url is empty. Paste a direct "
                "Hugging Face resolve/main URL or turn Enable LoRA OFF."
            )
        state_dict = self._load_or_download(
            lora_url.strip(),
            lora_save_name,
            _resolve_bearer_token(bearer_token, lora_url),
        )
        patched_model, _ = comfy.sd.load_lora_for_models(
            model, None, state_dict, strength_model, 0
        )
        return (patched_model,)

    @classmethod
    def IS_CHANGED(
        cls,
        model,
        enable_lora,
        lora_url,
        lora_save_name,
        bearer_token,
        strength_model,
    ):
        if not enable_lora:
            return "disabled"
        return f"{lora_url}|{lora_save_name}|{strength_model}"
        try:
            path = folder_paths.get_annotated_filepath(latent_file)
            return f"{path}:{os.stat(path).st_mtime_ns}"
        except Exception:
            return float("NaN")


NODE_CLASS_MAPPINGS = {
    "H3LatentUploadPath": H3LatentUploadPath,
    "H3OptionalLatentUploadLoader": H3OptionalLatentUploadLoader,
    "H3ExternalLoRASettings": H3ExternalLoRASettings,
    "H3SafeExternalLoRAApplyModel": H3SafeExternalLoRAApplyModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3LatentUploadPath": "H3 Latent Upload (Workflow)",
    "H3OptionalLatentUploadLoader": "H3 Optional Context Latent Upload + Load",
    "H3ExternalLoRASettings": "H3 External LoRA Settings (Safe)",
    "H3SafeExternalLoRAApplyModel": "H3 Safe External LoRA Apply (Model)",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
