"""Patch harbor 0.24.0 singularity.py for rootless apptainer without /etc/subuid (idempotent).

    python patches/patch_harbor_singularity.py <path/to/harbor/environments/singularity/singularity.py>

With a root-mapped fakeroot namespace, image files belong to unmapped UIDs, so the stock
`--writable-tmpfs` overlay cannot copy-up or rename lower-layer directories (EOVERFLOW /
EXDEV), which breaks apt/dpkg in both agents and verifiers. Instead, each image is unpacked
once into a sandbox directory owned by the invoking user, every trial gets its own
reflink copy (instant on XFS), and the container runs with `--writable`. The stock bootstrap
copy (bootstrap.stock.sh) is also staged for the patched bootstrap.sh fallback.
"""

import sys
from pathlib import Path

MARK = "# [qwen36-takehome patch]"

HELPER = f'''
    {MARK} per-trial writable sandbox copies
    async def _writable_trial_copy(self, sif_path: Path) -> Path:
        sandbox = sif_path.with_suffix(".sandbox")
        lock = open(str(sandbox) + ".lock", "w")
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: fcntl.flock(lock.fileno(), fcntl.LOCK_EX))
            if not sandbox.exists():
                tmp = Path(str(sandbox) + f".tmp.{{self.session_id}}")
                proc = await asyncio.create_subprocess_exec(
                    "apptainer", "build", "--sandbox", str(tmp), str(sif_path),
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                _, err = await proc.communicate()
                if proc.returncode != 0:
                    shutil.rmtree(tmp, ignore_errors=True)
                    raise RuntimeError(f"apptainer build --sandbox failed: {{err.decode(errors='replace')[-2000:]}}")
                tmp.rename(sandbox)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()
        trials = sif_path.parent / "trials"
        trials.mkdir(exist_ok=True)
        copy = trials / self.session_id
        shutil.rmtree(copy, ignore_errors=True)
        proc = await asyncio.create_subprocess_exec("cp", "-a", "--reflink=auto", str(sandbox), str(copy))
        if await proc.wait() != 0:
            raise RuntimeError(f"failed to copy sandbox for {{self.session_id}}")
        # --writable cannot create bind destinations; make every mount target exist.
        targets = [("/staging", True)] + [
            (m["target"], not Path(m["source"]).is_file())
            for m in (self._mounts or []) if m.get("type") == "bind"]
        for target, is_dir in targets:
            dest = copy / target.lstrip("/")
            if is_dir:
                dest.mkdir(parents=True, exist_ok=True)
            elif not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.touch()
        self._trial_copy = copy
        return copy

'''


def main(path):
    p = Path(path)
    s = p.read_text()
    if MARK in s:
        print("already patched:", p)
        return
    edits = [
        ('                "--writable-tmpfs",\n', '                "--writable",  ' + MARK + '\n'),
        ("        await self._start_server()\n\n        await self._upload_environment_dir_after_start()\n",
         "        self._sif_path = await self._writable_trial_copy(self._sif_path)  " + MARK + "\n"
         "        await self._start_server()\n\n        await self._upload_environment_dir_after_start()\n"),
        ("        bootstrap_script.chmod(0o755)\n",
         "        bootstrap_script.chmod(0o755)\n"
         "        shutil.copy(Path(__file__).parent / \"bootstrap.stock.sh\", self._staging_dir / \"bootstrap.stock.sh\")  " + MARK + "\n"
         "        (self._staging_dir / \"bootstrap.stock.sh\").chmod(0o755)\n"),
        ("    @override\n    async def stop(self, delete: bool) -> None:\n",
         HELPER + "    @override\n    async def stop(self, delete: bool) -> None:\n"
         "        try:\n            await self._stop_inner(delete)\n        finally:\n"
         "            if getattr(self, \"_trial_copy\", None):  " + MARK + "\n"
         "                shutil.rmtree(self._trial_copy, ignore_errors=True)\n\n"
         "    async def _stop_inner(self, delete: bool) -> None:\n"),
    ]
    for old, new in edits:
        if s.count(old) != 1:
            sys.exit(f"patch anchor not found exactly once: {old[:60]!r}")
        s = s.replace(old, new, 1)
    p.write_text(s)
    print("patched:", p)


if __name__ == "__main__":
    main(sys.argv[1])
