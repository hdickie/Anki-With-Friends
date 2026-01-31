# api/routes_analysis.py
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import os, subprocess, json, time
from pathlib import Path
from fastapi import APIRouter, HTTPException
from datetime import datetime
from fastapi.responses import JSONResponse
import sys
import logging
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent        # /app/api
TEMPLATES_DIR = BASE_DIR / "templates"            # /app/api/templates

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

ANALYSIS_DIR = Path(os.getenv("ANALYSIS_DIR", "/data"))
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = ANALYSIS_DIR / "zipf_vocab_analysis.json"
SCRIPT_PATH = Path(os.getenv("VOCAB_SCRIPT_PATH", "/app/scripts/vocab-analysis.py"))

def get_file_info(path: Path):
    if not path.exists():
        return None

    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
    }


@router.get("/analysis/vocab", response_class=HTMLResponse)
def vocab_page(request: Request):
    logger.warning("GET vocab_page: OUTPUT_JSON=%s exists=%s", OUTPUT_JSON, OUTPUT_JSON.exists())
    if OUTPUT_JSON.exists():
        logger.warning("GET vocab_page: stat size=%s mtime=%s", OUTPUT_JSON.stat().st_size, OUTPUT_JSON.stat().st_mtime)

    file_info = get_file_info(OUTPUT_JSON)
    logger.warning("GET vocab_page: file_info=%s", file_info)

    return templates.TemplateResponse(
        "analysis/vocab.html",
        {"request": request, "file_info": file_info},
    )

@router.post("/analysis/vocab/run")
def vocab_run():
    """
    Runs the analysis script. Script should write OUTPUT_JSON.
    Then redirect back to the UI page to display updated results.
    """

    # Basic “dev safety”: ensure script exists
    if not SCRIPT_PATH.exists():
        # Redirect with a simple marker; you can do proper flash messages later
        return RedirectResponse(url="/ui/your-analytics?error=missing_script", status_code=303)

    # Run it. If your script writes to OUTPUT_JSON, great.
    # If the script needs output path, pass it via env.
    env = os.environ.copy()
    env["ANALYSIS_DIR"] = str(ANALYSIS_DIR)
    env["OUTPUT_JSON"] = str(OUTPUT_JSON)

    logger.info("Vocab page file_info: %s", env["OUTPUT_JSON"])

    try:
        
        result = subprocess.run(
            ["python", str(SCRIPT_PATH)],
            cwd="/app",
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
            timeout=300,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return RedirectResponse(url="/ui/your-analytics?error=timeout", status_code=303)

    # If script failed, redirect with error and stash a small log file for inspection.
    if result.returncode != 0:
        log_path = ANALYSIS_DIR / "zipf_vocab_analysis_last_error.log"
        log_path.write_text(
            f"returncode={result.returncode}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n",
            encoding="utf-8",
        )
        return RedirectResponse(url="/ui/your-analytics?error=script_failed", status_code=303)

    # Ensure output exists
    if not OUTPUT_JSON.exists():
        return RedirectResponse(url="/ui/your-analytics?error=no_output", status_code=303)

    return RedirectResponse(url="/ui/your-analytics?ok=1", status_code=303)


def _file_status(path: Path):
    if not path.exists():
        return {"exists": False}

    st = path.stat()
    return {
        "exists": True,
        "path": str(path),
        "size_bytes": st.st_size,
        "modified": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
    }


@router.get("/analysis/vocab/status")
def vocab_status():
    return JSONResponse(_file_status(OUTPUT_JSON))

@router.get("/analysis/vocab/json")
def vocab_json():
    analysis_dir = Path(os.getenv("ANALYSIS_DIR", "/data"))
    path = analysis_dir / "zipf_vocab_analysis.json"
    if not path.exists():
        # 404 makes the UI logic clearer too
        raise HTTPException(status_code=404, detail="zipf_vocab_analysis.json not found")

    return JSONResponse(content=json.loads(path.read_text(encoding="utf-8")))