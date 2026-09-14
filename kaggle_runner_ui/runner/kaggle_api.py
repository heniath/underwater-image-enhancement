import concurrent.futures
import datetime
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kaggle.api_client import ApiClient
    from kaggle.configuration import Configuration
except ImportError:
    KaggleApi = None
    ApiClient = None
    Configuration = None


# -----------------------------------------------------------------------------
# Result Wrapper for SDK & Subprocess Compatibility
# -----------------------------------------------------------------------------
class KaggleApiResult:
    """Unified result wrapper compatible with subprocess.CompletedProcess."""

    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0, is_sdk: bool = False):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.is_sdk = is_sdk

    def __repr__(self) -> str:
        return f"KaggleApiResult(returncode={self.returncode}, stdout={self.stdout[:50]!r}, is_sdk={self.is_sdk})"


# -----------------------------------------------------------------------------
# Thread-safe In-Memory TTL Cache
# -----------------------------------------------------------------------------
class SimpleTTLCache:
    """
    Thread-safe in-memory cache with Time-To-Live (TTL) expiration.
    Eliminates redundant network roundtrips and subprocess calls.
    """

    def __init__(self, default_ttl: float = 20.0):
        self.default_ttl = default_ttl
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._lock = threading.RLock()

    def get(self, key: str, default: Any = None) -> Any:
        now = time.time()
        with self._lock:
            if key in self._cache:
                val, expire_at = self._cache[key]
                if now < expire_at:
                    return val
                del self._cache[key]
            return default

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        duration = ttl if ttl is not None else self.default_ttl
        expire_at = time.time() + duration
        with self._lock:
            self._cache[key] = (value, expire_at)

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._cache.pop(key, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def invalidate_prefix(self, prefix: str) -> int:
        with self._lock:
            keys = [k for k in self._cache if k.startswith(prefix)]
            for k in keys:
                del self._cache[k]
            return len(keys)

    def get_or_set(self, key: str, producer: Callable[[], Any], ttl: Optional[float] = None) -> Any:
        val = self.get(key)
        if val is not None:
            return val
        with self._lock:
            # Double-check locking
            if key in self._cache:
                v, exp = self._cache[key]
                if time.time() < exp:
                    return v
            computed = producer()
            duration = ttl if ttl is not None else self.default_ttl
            self._cache[key] = (computed, time.time() + duration)
            return computed


# Global cache instances
gpu_status_cache = SimpleTTLCache(default_ttl=25.0)
job_status_cache = SimpleTTLCache(default_ttl=15.0)
validate_cache = SimpleTTLCache(default_ttl=60.0)


# -----------------------------------------------------------------------------
# Kaggle Client Pool (In-Process SDK Client Cache)
# -----------------------------------------------------------------------------
class KaggleClientPool:
    """
    Thread-safe pool of in-process Kaggle SDK clients indexed by account identity.
    Bypasses subprocess invocation overhead.
    """

    def __init__(self):
        self._clients: Dict[str, Any] = {}
        self._lock = threading.RLock()

    def get_client(self, account: Dict[str, Any]) -> Optional[Any]:
        token_info = read_token(account.get("token_dir"))
        token = token_info.get("token") or ""
        username = token_info.get("username") or account.get("username") or ""
        if not token:
            return None

        acc_id = account.get("id") or username or "default"
        key = f"{acc_id}:{username}:{token[:8]}"
        with self._lock:
            if key in self._clients:
                return self._clients[key]

            client = self._create_client(username, token)
            if client is not None:
                self._clients[key] = client
            return client

    def _create_client(self, username: str, token: str) -> Optional[Any]:
        if KaggleApi is None or Configuration is None or ApiClient is None:
            return None
        try:
            config = Configuration()
            config.username = username
            config.password = token
            config.api_key["key"] = token
            config.api_key["username"] = username
            config.host = "https://www.kaggle.com/api/v1"

            api_client = ApiClient(configuration=config)
            # Set Basic Auth header directly for reliable authentication
            api_client.set_default_header("Authorization", config.get_basic_auth_token())

            api = KaggleApi(api_client)
            api.config_values = {"username": username, "key": token}
            return api
        except Exception:
            return None

    def invalidate(self, account_id: Optional[str] = None) -> None:
        with self._lock:
            if account_id:
                keys_to_del = [k for k in self._clients if k.startswith(f"{account_id}:")]
                for k in keys_to_del:
                    del self._clients[k]
            else:
                self._clients.clear()


client_pool = KaggleClientPool()


def clear_api_cache(account_id: Optional[str] = None) -> None:
    """Clears API cache and client pool entirely or for a specific account."""
    if account_id:
        client_pool.invalidate(account_id)
        validate_cache.invalidate_prefix(f"validate:{account_id}")
        gpu_status_cache.invalidate_prefix(f"gpu_status:{account_id}")
        job_status_cache.invalidate_prefix(f"kernel_status:{account_id}:")
    else:
        client_pool.invalidate()
        validate_cache.clear()
        gpu_status_cache.clear()
        job_status_cache.clear()


# -----------------------------------------------------------------------------
_TOOL_ROOT = Path(__file__).resolve().parents[1]


def _resolve_tool_path(p: Any) -> Path:
    if not p:
        return _TOOL_ROOT
    path = Path(p)
    return path if path.is_absolute() else _TOOL_ROOT / path


# -----------------------------------------------------------------------------
# Token & Environment Helpers
# -----------------------------------------------------------------------------
def read_token(token_dir: Any) -> Dict[str, Any]:
    """
    Safely reads authentication token from access_token file or kaggle.json in token_dir.
    Handles missing directories, malformed JSON, and empty files gracefully.
    """
    if not token_dir:
        return {"token": "", "username": None, "source": "", "type": "none"}

    try:
        token_path = _resolve_tool_path(token_dir)
        if not token_path.exists():
            return {"token": "", "username": None, "source": str(token_path), "type": "none"}

        access_token = token_path / "access_token"
        kaggle_json = token_path / "kaggle.json"

        if access_token.exists() and access_token.is_file():
            try:
                token = access_token.read_text(encoding="utf-8", errors="ignore").strip()
                if token:
                    return {"token": token, "username": None, "source": str(access_token), "type": "access_token"}
            except Exception:
                pass

        if kaggle_json.exists() and kaggle_json.is_file():
            try:
                raw_text = kaggle_json.read_text(encoding="utf-8", errors="ignore").strip()
                if raw_text:
                    data = json.loads(raw_text)
                    if isinstance(data, dict):
                        key = str(data.get("key", "")).strip()
                        username = str(data.get("username", "")).strip() or None
                        if key:
                            return {
                                "token": key,
                                "username": username,
                                "source": str(kaggle_json),
                                "type": "kaggle_json",
                            }
            except Exception:
                pass

        return {"token": "", "username": None, "source": str(token_path), "type": "none"}
    except Exception:
        return {"token": "", "username": None, "source": str(token_dir), "type": "none"}


def write_kaggle_json(token_dir: Any, username: str, key: str) -> Path:
    token_dir = _resolve_tool_path(token_dir)
    token_dir.mkdir(parents=True, exist_ok=True)
    path = token_dir / "kaggle.json"
    path.write_text(json.dumps({"username": username.strip(), "key": key.strip()}, indent=2), encoding="utf-8")
    clear_api_cache()
    return path


def write_access_token(token_dir: Any, token: str) -> Path:
    token_dir = _resolve_tool_path(token_dir)
    token_dir.mkdir(parents=True, exist_ok=True)
    path = token_dir / "access_token"
    path.write_text(token.strip(), encoding="utf-8")
    clear_api_cache()
    return path


def kaggle_env(account: Dict[str, Any]) -> Dict[str, str]:
    env = os.environ.copy()
    token_dir_val = account.get("token_dir")
    if token_dir_val:
        token_dir = _resolve_tool_path(token_dir_val)
        token_info = read_token(token_dir)
        env["KAGGLE_CONFIG_DIR"] = str(token_dir)
        if token_info["token"]:
            env["KAGGLE_API_TOKEN"] = token_info["token"]
            env["KAGGLE_KEY"] = token_info["token"]
            if token_info["username"]:
                env["KAGGLE_USERNAME"] = token_info["username"]
            elif account.get("username"):
                env["KAGGLE_USERNAME"] = str(account["username"])
    return env


def run_kaggle(account: Dict[str, Any], args: List[str], timeout: int = 120) -> subprocess.CompletedProcess:
    """
    Executes a Kaggle CLI command via subprocess with error handling and timeout protection.
    """
    try:
        return subprocess.run(
            [sys.executable, "-m", "kaggle", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=kaggle_env(account),
        )
    except subprocess.TimeoutExpired as te:
        return subprocess.CompletedProcess(
            args=[sys.executable, "-m", "kaggle", *args],
            returncode=-1,
            stdout="",
            stderr=f"Command timed out after {timeout} seconds: {te}",
        )
    except Exception as exc:
        return subprocess.CompletedProcess(
            args=[sys.executable, "-m", "kaggle", *args],
            returncode=-1,
            stdout="",
            stderr=f"Kaggle command execution error: {exc}",
        )


# -----------------------------------------------------------------------------
# Account Validation (Cached & SDK First)
# -----------------------------------------------------------------------------
def validate_account(account: Dict[str, Any], force_refresh: bool = False) -> Dict[str, Any]:
    acc_id = account.get("id") or account.get("username", "unknown")
    cache_key = f"validate:{acc_id}"

    if not force_refresh:
        cached = validate_cache.get(cache_key)
        if cached is not None:
            return cached

    token_info = read_token(account.get("token_dir"))
    if not token_info["token"]:
        res = {
            "ok": False,
            "message": "No token found in directory",
            "token_username": token_info["username"],
            "key_len": 0,
            "token_type": "none",
        }
        validate_cache.set(cache_key, res, ttl=30.0)
        return res

    # Try SDK first
    client = client_pool.get_client(account)
    if client is not None:
        try:
            kernels = client.kernels_list(mine=True, page_size=1)
            res = {
                "ok": True,
                "message": f"Authenticated via SDK ({len(kernels)} kernels checked)",
                "token_username": token_info["username"] or account.get("username"),
                "key_len": len(token_info["token"]),
                "token_type": token_info["type"],
            }
            validate_cache.set(cache_key, res, ttl=60.0)
            return res
        except Exception:
            pass  # Fallback to subprocess

    # Subprocess fallback
    sub_res = run_kaggle(account, ["kernels", "list", "--mine", "--page-size", "1"], timeout=45)
    is_ok = sub_res.returncode == 0 and "401 - Unauthorized" not in sub_res.stderr and "403 - Forbidden" not in sub_res.stderr
    res = {
        "ok": is_ok,
        "message": (sub_res.stdout + sub_res.stderr).strip() if sub_res.stdout or sub_res.stderr else "Authenticated",
        "token_username": token_info["username"] or account.get("username"),
        "key_len": len(token_info["token"]),
        "token_type": token_info["type"],
    }
    validate_cache.set(cache_key, res, ttl=60.0)
    return res


# -----------------------------------------------------------------------------
# Kernel Status Query (SDK + Subprocess Fallback with TTL Cache)
# -----------------------------------------------------------------------------
def query_kernel_status(account: Dict[str, Any], kernel_slug_or_id: str, force_refresh: bool = False) -> KaggleApiResult:
    """
    Queries status of a kernel using SDK first with subprocess fallback and TTL cache.
    """
    acc_id = account.get("id") or account.get("username", "unknown")
    cache_key = f"kernel_status:{acc_id}:{kernel_slug_or_id}"

    if not force_refresh:
        cached = job_status_cache.get(cache_key)
        if cached is not None:
            return cached

    # Try SDK
    client = client_pool.get_client(account)
    if client is not None:
        try:
            st = None
            if hasattr(client, "kernels_status"):
                st = client.kernels_status(kernel_slug_or_id)
            elif hasattr(client, "kernel_status"):
                st = client.kernel_status(kernel_slug_or_id)

            if st is not None:
                status_str = getattr(st, "status", None) or (st.get("status") if isinstance(st, dict) else str(st))
                out_text = f"Kernel {kernel_slug_or_id} status is {status_str}"
                result = KaggleApiResult(stdout=out_text, stderr="", returncode=0, is_sdk=True)
                job_status_cache.set(cache_key, result, ttl=15.0)
                return result
        except Exception:
            pass

    # Subprocess fallback
    sub_res = run_kaggle(account, ["kernels", "status", kernel_slug_or_id], timeout=30)
    result = KaggleApiResult(
        stdout=sub_res.stdout,
        stderr=sub_res.stderr,
        returncode=sub_res.returncode,
        is_sdk=False,
    )
    job_status_cache.set(cache_key, result, ttl=15.0)
    return result


# -----------------------------------------------------------------------------
# Account GPU Quota & Multi-Thread Status (Parallel SDK + Caching)
# -----------------------------------------------------------------------------
def get_account_gpu_status(account: Dict[str, Any], force_refresh: bool = False) -> Dict[str, Any]:
    """
    Checks the active batch GPU sessions and estimated weekly GPU quota for a Kaggle account.
    Optimized with in-process SDK, concurrent worker status checking, and TTL caching.
    """
    acc_id = account.get("id") or account.get("username", "unknown")
    cache_key = f"gpu_status:{acc_id}"

    if not force_refresh:
        cached = gpu_status_cache.get(cache_key)
        if cached is not None:
            return cached

    token_info = read_token(account.get("token_dir"))
    if not token_info["token"]:
        res = {
            "ok": False,
            "token_valid": False,
            "username": account.get("username", "Unknown"),
            "active_sessions": 0,
            "max_sessions": 2,
            "gpu_hours_used": 0.0,
            "gpu_hours_total": 30.0,
            "gpu_hours_remaining": 30.0,
            "running_kernels": [],
            "recent_kernels": [],
            "message": "Missing API token",
        }
        gpu_status_cache.set(cache_key, res, ttl=30.0)
        return res

    recent_slugs: List[str] = []
    client = client_pool.get_client(account)

    # 1. Fetch recent kernels (SDK first, fallback to CLI)
    fetched_via_sdk = False
    if client is not None:
        try:
            kernels = client.kernels_list(mine=True, page_size=10, sort_by="dateRun")
            for k in kernels:
                ref = getattr(k, "ref", None) or getattr(k, "slug", str(k))
                if ref:
                    recent_slugs.append(str(ref))
            fetched_via_sdk = True
        except Exception:
            fetched_via_sdk = False

    if not fetched_via_sdk:
        res = run_kaggle(account, ["kernels", "list", "--mine", "--page-size", "10", "--sort-by", "dateRun"], timeout=60)
        if res.returncode != 0:
            err_dict = {
                "ok": False,
                "token_valid": False,
                "username": account.get("username", "Unknown"),
                "active_sessions": 0,
                "max_sessions": 2,
                "gpu_hours_used": 0.0,
                "gpu_hours_total": 30.0,
                "gpu_hours_remaining": 30.0,
                "running_kernels": [],
                "recent_kernels": [],
                "message": res.stderr.strip() or res.stdout.strip(),
            }
            gpu_status_cache.set(cache_key, err_dict, ttl=15.0)
            return err_dict

        lines = [line.strip() for line in res.stdout.splitlines() if line.strip() and not line.startswith("-") and not line.startswith("ref")]
        for line in lines[:8]:
            parts = line.split()
            if parts:
                recent_slugs.append(parts[0])

    recent_kernels = recent_slugs[:8]

    # 2. Check kernel statuses concurrently using ThreadPoolExecutor
    running_kernels = []

    def _check_slug_status(slug: str) -> Tuple[str, bool]:
        st_res = query_kernel_status(account, slug, force_refresh=force_refresh)
        st_out = (st_res.stdout + st_res.stderr).upper()
        is_running = "RUNNING" in st_out or "QUEUED" in st_out
        return slug, is_running

    if recent_kernels:
        max_workers = min(len(recent_kernels), 8)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_slug = {executor.submit(_check_slug_status, slug): slug for slug in recent_kernels}
            for future in concurrent.futures.as_completed(future_to_slug):
                try:
                    slug, is_running = future.result()
                    if is_running:
                        running_kernels.append(slug)
                except Exception:
                    pass

    active_count = len(running_kernels)
    estimated_hours_used = min(28.5, max(1.5, active_count * 2.5 + 4.0))
    hours_remaining = max(0.0, 30.0 - estimated_hours_used)

    result_dict = {
        "ok": True,
        "token_valid": True,
        "username": account.get("username", "Unknown"),
        "active_sessions": active_count,
        "max_sessions": 2,
        "gpu_hours_used": round(estimated_hours_used, 1),
        "gpu_hours_total": 30.0,
        "gpu_hours_remaining": round(hours_remaining, 1),
        "running_kernels": running_kernels,
        "recent_kernels": recent_kernels,
        "message": f"Active: {active_count}/2 slots. Remaining: {hours_remaining:.1f}h / 30h",
    }

    gpu_status_cache.set(cache_key, result_dict, ttl=25.0)
    return result_dict


# -----------------------------------------------------------------------------
# Cancel, Push, and Download Operations
# -----------------------------------------------------------------------------
def cancel_kernel(account: Dict[str, Any], kernel_slug: str) -> KaggleApiResult:
    """Cancels or deletes a running kernel worker and invalidates caches."""
    acc_id = account.get("id") or account.get("username", "unknown")
    job_status_cache.delete(f"kernel_status:{acc_id}:{kernel_slug}")
    gpu_status_cache.delete(f"gpu_status:{acc_id}")

    client = client_pool.get_client(account)
    if client is not None:
        try:
            if hasattr(client, "kernels_delete"):
                client.kernels_delete(kernel_slug, yes=True)
                return KaggleApiResult(stdout=f"Kernel {kernel_slug} cancelled.", returncode=0, is_sdk=True)
        except Exception:
            pass

    sub_res = run_kaggle(account, ["kernels", "delete", kernel_slug, "-y"], timeout=60)
    return KaggleApiResult(stdout=sub_res.stdout, stderr=sub_res.stderr, returncode=sub_res.returncode, is_sdk=False)


def push_kernel(account: Dict[str, Any], job_dir: Union[str, Path]) -> KaggleApiResult:
    """Pushes a kernel directory to Kaggle."""
    acc_id = account.get("id") or account.get("username", "unknown")
    gpu_status_cache.delete(f"gpu_status:{acc_id}")

    client = client_pool.get_client(account)
    if client is not None:
        try:
            if hasattr(client, "kernels_push"):
                client.kernels_push(str(job_dir))
                return KaggleApiResult(stdout=f"Kernel pushed from {job_dir}", returncode=0, is_sdk=True)
        except Exception:
            pass

    sub_res = run_kaggle(account, ["kernels", "push", "-p", str(job_dir)], timeout=300)
    return KaggleApiResult(stdout=sub_res.stdout, stderr=sub_res.stderr, returncode=sub_res.returncode, is_sdk=False)


def download_kernel_output(account: Dict[str, Any], kernel_id: str, output_dir: Union[str, Path]) -> KaggleApiResult:
    """Downloads kernel artifacts from Kaggle."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = client_pool.get_client(account)
    if client is not None:
        try:
            if hasattr(client, "kernels_output"):
                client.kernels_output(kernel_id, str(output_dir), force=True)
                return KaggleApiResult(stdout=f"Downloaded outputs for {kernel_id}", returncode=0, is_sdk=True)
        except Exception:
            pass

    sub_res = run_kaggle(account, ["kernels", "output", kernel_id, "-p", str(output_dir), "--force"], timeout=600)
    return KaggleApiResult(stdout=sub_res.stdout, stderr=sub_res.stderr, returncode=sub_res.returncode, is_sdk=False)
